import asyncio
from collections.abc import AsyncIterator
from os import getenv
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field
from pydantic_ai import AgentRunResultEvent
from pydantic_ai.exceptions import AgentRunError, ModelHTTPError
from pydantic_ai.messages import (
    FunctionToolCallEvent,
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

from app.agents import Approach, PageRead, RunDeps, build_agent
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
async def config() -> dict[str, str | bool]:
    return {"model": HF_MODEL, "ready": bool(configured_token())}


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
            async with agent.run_stream_events(
                body.prompt.strip(),
                deps=deps,
                usage_limits=UsageLimits(request_limit=10, tool_calls_limit=12, output_tokens_limit=4000),
            ) as events:
                async for event in events:
                    if isinstance(event, FunctionToolCallEvent):
                        part = event.part
                        yield ServerSentEvent(event="tool", data={
                            "name": part.tool_name,
                            "code": part.code if isinstance(part, RunCodeCallPart) else None,
                        })
                    elif isinstance(event, PageRead):
                        yield ServerSentEvent(event="page", data={"cursor": event.cursor, "items": event.items})
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
