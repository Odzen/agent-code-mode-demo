import asyncio
from collections.abc import AsyncIterator
from os import getenv
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import FastAPI, Header, Request
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
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.code_mode import RunCodeCallPart

from app.agents import Approach, PageRead, RunDeps, build_agent
from app.sample import EXAMPLE_PROMPT, example_model
from app.tickets import router as tickets_router

app = FastAPI(title="Agent Run Lab")
app.include_router(tickets_router)


class RunInput(BaseModel):
    approach: Approach
    prompt: str = Field(min_length=1, max_length=2000)
    mode: Literal["example", "live"] = "example"
    model: Literal["deepseek-flash", "deepseek-v4-pro"] = "deepseek-flash"


@app.post("/api/run", response_class=EventSourceResponse)
async def run(
    body: RunInput,
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> AsyncIterator[ServerSentEvent]:
    if body.mode == "example" and body.prompt.strip() != EXAMPLE_PROMPT:
        yield ServerSentEvent(event="error", data="Use the example question or switch to Live.")
        return
    key = None
    if body.mode == "live":
        if authorization and authorization.startswith("Bearer "):
            key = authorization.removeprefix("Bearer ").strip()
        else:
            key = getenv("DEEPSEEK_API_KEY")
        if not key:
            yield ServerSentEvent(event="error", data="Enter a DeepSeek API key to run live.")
            return

    try:
        async with asyncio.timeout(60), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=request.app),
            base_url="http://tickets.local",
        ) as client:
            model = (
                OpenAIChatModel(body.model, provider=DeepSeekProvider(api_key=key))
                if key
                else example_model()
            )
            agent = build_agent(model, body.approach)
            deps = RunDeps(tickets=client)
            async with agent.run_stream_events(
                body.prompt.strip(),
                deps=deps,
                usage_limits=UsageLimits(request_limit=10, tool_calls_limit=12, output_tokens_limit=4000),
                model_settings=OpenAIChatModelSettings(thinking=False) if key else None,
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
                            "input_tokens": usage.input_tokens if key else None,
                        })
    except ModelHTTPError as error:
        if error.status_code in {401, 403}:
            message = "DeepSeek rejected the API key. Check the key and try again."
        elif error.status_code == 402:
            message = "The DeepSeek account needs an API balance."
        elif error.status_code == 429:
            message = "DeepSeek is rate limiting requests. Try again in a moment."
        else:
            message = "DeepSeek could not complete the request. Try again."
        yield ServerSentEvent(event="error", data=message)
    except (AgentRunError, httpx.HTTPError, TimeoutError):
        yield ServerSentEvent(event="error", data="The run stopped before it could finish. Try again.")


dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
app.frontend("/", directory=dist, check_dir=False, fallback=None)
