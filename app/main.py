import asyncio
from collections.abc import AsyncIterator
from os import getenv
from pathlib import Path
from time import perf_counter
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python
from pydantic_ai import AgentRunResultEvent
from pydantic_ai.exceptions import AgentRunError, ModelHTTPError
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolReturnPart,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.code_mode import RunCodeCallPart

from app.agents import DATA_TOOL_NAMES, Approach, PageRead, RunDeps, ToolActivity, build_agent
from app.tickets import router as tickets_router

HF_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017:novita"
HF_ROUTER = "https://router.huggingface.co/v1"

app = FastAPI(title="Agent Run Lab")
app.include_router(tickets_router)


class RunInput(BaseModel):
    approach: Approach
    prompt: str = Field(min_length=1, max_length=2000)


def configured_token() -> str:
    return getenv("HF_TOKEN", "").strip()


@app.get("/api/config")
async def config() -> dict[str, str | bool | list[str]]:
    return {"model": HF_MODEL, "ready": bool(configured_token()), "tools": DATA_TOOL_NAMES}


def require_token() -> str:
    token = configured_token()
    if not token:
        raise HTTPException(503, "The demo is waiting for its Hugging Face token.")
    return token


@app.post("/api/run", response_class=EventSourceResponse)
async def run(
    body: RunInput,
    request: Request,
    token: Annotated[str, Depends(require_token)],
) -> AsyncIterator[ServerSentEvent]:
    started = perf_counter()
    try:
        async with asyncio.timeout(60), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=request.app),
            base_url="http://tickets.local",
        ) as client:
            model = OpenAIChatModel(
                HF_MODEL,
                provider=OpenAIProvider(base_url=HF_ROUTER, api_key=token),
                profile=OpenAIModelProfile(
                    openai_chat_thinking_field="reasoning_content",
                    openai_chat_send_back_thinking_parts="field",
                    openai_chat_thinking_requires_field_on_tool_calls=True,
                ),
            )
            agent = build_agent(model, body.approach)
            deps = RunDeps(tickets=client)
            script_call_ids: set[str] = set()
            async with agent.run_stream_events(
                body.prompt.strip(),
                deps=deps,
                usage_limits=UsageLimits(request_limit=10, tool_calls_limit=12, output_tokens_limit=4000),
            ) as events:
                async for event in events:
                    if isinstance(event, FunctionToolCallEvent):
                        part = event.part
                        if isinstance(part, RunCodeCallPart):
                            script_call_ids.add(part.tool_call_id)
                        yield ServerSentEvent(event="tool", data={
                            "call_id": part.tool_call_id,
                            "name": part.tool_name,
                            "sandbox": isinstance(part, RunCodeCallPart),
                            "code": part.code if isinstance(part, RunCodeCallPart) else None,
                        })
                    elif isinstance(event, FunctionToolResultEvent):
                        part = event.part
                        outcome = part.outcome
                        payload = {"output": to_jsonable_python(part.content)} if outcome == "success" else {}
                        output = payload.get("output")
                        if part.tool_name in DATA_TOOL_NAMES and isinstance(output, dict) and "error" in output:
                            outcome = "failed"
                        if outcome == "retried" and part.tool_call_id in script_call_ids:
                            reason = part.model_response_str(wrap_if_error=False).partition(":")[0]
                            if reason in {"Syntax error in code", "Type error in code", "Runtime error"}:
                                payload["detail"] = reason
                        yield ServerSentEvent(event="tool_result", data={
                            "call_id": part.tool_call_id,
                            "name": part.tool_name,
                            "outcome": outcome,
                            **payload,
                        })
                    elif isinstance(event, (ToolActivity, PageRead)):
                        call_id = event.tool_call_id
                        # Code Mode names nested calls <run_code call id>__<index>.
                        parent_id, _, index = (call_id or "").rpartition("__")
                        trace_ids = {
                            "call_id": call_id,
                            "parent_id": parent_id if index.isdecimal() and parent_id in script_call_ids else None,
                        }
                        if trace_ids["parent_id"] is not None:
                            trace_ids["output"] = to_jsonable_python(event.output)
                            trace_ids["outcome"] = (
                                "failed" if isinstance(event.output, dict) and "error" in event.output else "success"
                            )
                        if isinstance(event, ToolActivity):
                            yield ServerSentEvent(event="activity", data={
                                **trace_ids, "name": event.label, "detail": event.detail,
                            })
                        else:
                            yield ServerSentEvent(event="page", data={
                                **trace_ids, "cursor": event.cursor, "items": event.items,
                            })
                    elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                        yield ServerSentEvent(event="text", data=event.part.content)
                    elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                        yield ServerSentEvent(event="text", data=event.delta.content_delta)
                    elif isinstance(event, AgentRunResultEvent):
                        result = event.result
                        tool_bytes = sum(
                            len(part.model_response_str().encode("utf-8"))
                            for message in result.new_messages()
                            if isinstance(message, ModelRequest)
                            for part in message.parts
                            if isinstance(part, ToolReturnPart)
                        )
                        usage = result.usage
                        yield ServerSentEvent(event="done", data={
                            "answer": result.output,
                            "model_requests": usage.requests,
                            "pages_read": deps.pages_read,
                            "tool_output_bytes": tool_bytes,
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "elapsed_ms": round((perf_counter() - started) * 1000),
                        })
    except ModelHTTPError as error:
        if error.status_code in {401, 403}:
            message = "Hugging Face rejected the demo’s credentials."
        elif error.status_code == 402:
            message = "The Hugging Face account needs an inference balance."
        elif error.status_code == 404:
            message = "The configured model is unavailable from this inference provider."
        elif error.status_code == 429:
            message = "Hugging Face is rate limiting requests. Try again in a moment."
        else:
            message = "Hugging Face could not complete the request. Try again."
        yield ServerSentEvent(event="error", data=message)
    except (AgentRunError, httpx.HTTPError, TimeoutError):
        yield ServerSentEvent(event="error", data="The run stopped before it could finish. Try again.")


dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
app.frontend("/", directory=dist, check_dir=False, fallback=None)
