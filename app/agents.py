from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic_ai import Agent, CustomEvent, RunContext
from pydantic_ai.models import Model
from pydantic_ai_harness import CodeMode

from app.tickets import TicketPage

Approach = Literal["tool-calling", "code-mode"]
INSTRUCTIONS = (
    "Answer questions about the synthetic overdue ticket dataset. "
    "Follow pagination until the last page when answering about the whole dataset. "
    "Report the results concisely and state if the data you checked is incomplete."
)


@dataclass
class RunDeps:
    tickets: httpx.AsyncClient
    pages_read: int = 0


@dataclass(kw_only=True)
class PageRead(CustomEvent):
    cursor: int | None
    items: int


async def list_overdue_tickets(ctx: RunContext[RunDeps], cursor: int | None = None) -> dict:
    """Read one page of overdue tickets. Pass the previous next_cursor for the next page."""
    response = await ctx.deps.tickets.get("/api/tickets", params={"cursor": cursor or 0})
    response.raise_for_status()
    page = TicketPage.model_validate(response.json())
    ctx.deps.pages_read += 1
    await ctx.emit(PageRead(cursor=cursor, items=len(page.items)))
    return page.model_dump()


def build_agent(model: Model, approach: Approach) -> Agent[RunDeps, str]:
    capabilities = []
    if approach == "code-mode":
        capabilities.append(
            CodeMode(
                tools=["list_overdue_tickets"],
                max_tool_calls=12,
                resource_limits={"max_duration_secs": 3, "max_memory": 64 * 1024 * 1024},
            )
        )
    return Agent(
        model,
        deps_type=RunDeps,
        instructions=INSTRUCTIONS,
        tools=[list_overdue_tickets],
        capabilities=capabilities,
    )
