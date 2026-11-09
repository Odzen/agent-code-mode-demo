import re
from dataclasses import dataclass
from typing import Literal, TypedDict

import httpx
from pydantic_ai import Agent, CustomEvent, RunContext
from pydantic_ai.models import Model
from pydantic_ai_harness import CodeMode

from app.tickets import CUSTOMERS, CustomerProfile, TicketPage

Approach = Literal["tool-calling", "code-mode"]
ISSUE_REPO = "pydantic/pydantic-ai"
INSTRUCTIONS = (
    "Answer questions about the synthetic overdue ticket dataset. "
    "Follow pagination until the last page when answering about the whole dataset. "
    "Use the customer profile tool when a question asks for plans or SLAs. "
    "Use public issue search for technical references, cite the returned links, and keep "
    "public findings separate from the synthetic tickets. "
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


@dataclass(kw_only=True)
class ToolActivity(CustomEvent):
    label: str
    detail: str


class ToolError(TypedDict):
    error: str


class PublicIssue(TypedDict):
    title: str
    url: str
    state: str


class IssueSearchResult(TypedDict):
    issues: list[PublicIssue]


async def list_overdue_tickets(ctx: RunContext[RunDeps], cursor: int | None = None) -> TicketPage:
    """Read a page of overdue tickets. Start with cursor=None, then follow next_cursor until None."""
    response = await ctx.deps.tickets.get("/api/tickets", params={"cursor": cursor or 0})
    response.raise_for_status()
    page = TicketPage.model_validate(response.json())
    ctx.deps.pages_read += 1
    await ctx.emit(PageRead(cursor=cursor, items=len(page.items)))
    return page


async def get_customer_profile(ctx: RunContext[RunDeps], customer: str) -> CustomerProfile | ToolError:
    """Look up a synthetic customer by name and return their plan and response SLA in hours."""
    name = customer.strip().casefold()
    for profile, _ in CUSTOMERS:
        if profile.name.casefold() == name:
            await ctx.emit(ToolActivity(label="get_customer_profile", detail=profile.name))
            return profile
    return {"error": "Customer not found."}


async def search_public_issues(ctx: RunContext[RunDeps], query: str) -> IssueSearchResult | ToolError:
    """Search public Pydantic AI issues on GitHub using technical keywords. Returns titles and links."""
    keywords = " ".join(re.findall(r"[\w-]+", query[:120]))
    if not keywords:
        return {"error": "Provide technical search keywords."}

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.get(
                "https://api.github.com/search/issues",
                params={"q": f"{keywords} repo:{ISSUE_REPO} is:issue", "per_page": 3},
                headers={"Accept": "application/vnd.github+json", "User-Agent": "agent-code-mode-demo"},
            )
            response.raise_for_status()
            items = response.json()["items"]
    except (httpx.HTTPError, ValueError, KeyError):
        await ctx.emit(ToolActivity(label="search_public_issues", detail="GitHub unavailable"))
        return {"error": "Public issue search is unavailable. Continue with any local data."}

    issues: list[PublicIssue] = [
        {"title": item["title"], "url": item["html_url"], "state": item["state"]}
        for item in items
        if item["html_url"].startswith(f"https://github.com/{ISSUE_REPO}/issues/")
    ]
    await ctx.emit(ToolActivity(label="search_public_issues", detail=f"{len(issues)} GitHub results"))
    return {"issues": issues}


DATA_TOOLS = (list_overdue_tickets, get_customer_profile, search_public_issues)
DATA_TOOL_NAMES = [tool.__name__ for tool in DATA_TOOLS]


def build_agent(model: Model, approach: Approach) -> Agent[RunDeps, str]:
    capabilities = []
    instructions = INSTRUCTIONS
    if approach == "code-mode":
        instructions += (
            " When the task needs several dependent tool calls, put them in one run_code "
            "program where possible. For pagination, follow next_cursor in Python until it "
            "is null, then return only the data needed to answer. Use the documented return shape "
            "to write the program without exploratory calls."
        )
        capabilities.append(
            CodeMode(
                tools=DATA_TOOL_NAMES,
                max_tool_calls=12,
                resource_limits={"max_duration_secs": 3, "max_memory": 64 * 1024 * 1024},
            )
        )
    return Agent(
        model,
        deps_type=RunDeps,
        instructions=instructions,
        tools=DATA_TOOLS,
        capabilities=capabilities,
    )
