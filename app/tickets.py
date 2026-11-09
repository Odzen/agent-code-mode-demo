from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api")
PAGE_SIZE = 6


class Ticket(BaseModel):
    id: str
    customer: str
    subject: str


class TicketPage(BaseModel):
    items: list[Ticket]
    next_cursor: int | None


class CustomerProfile(BaseModel):
    name: str
    plan: str
    response_sla_hours: int


CUSTOMERS = (
    (CustomerProfile(name="Aster Labs", plan="Enterprise", response_sla_hours=4), 6),
    (CustomerProfile(name="Ember Works", plan="Pro", response_sla_hours=12), 5),
    (CustomerProfile(name="Juniper Studio", plan="Enterprise", response_sla_hours=4), 3),
    (CustomerProfile(name="Signal House", plan="Pro", response_sla_hours=12), 2),
    (CustomerProfile(name="Cedar Systems", plan="Starter", response_sla_hours=24), 1),
    (CustomerProfile(name="Northstar Print", plan="Starter", response_sla_hours=24), 1),
)
SUBJECTS = (
    "Invoice totals do not match the account summary",
    "Password reset email has not arrived",
    "Export remains pending after the nightly sync",
)
TICKETS = tuple(
    Ticket(
        id=f"T-{index + 101}",
        customer=customer,
        subject=SUBJECTS[index % len(SUBJECTS)],
    )
    for index, customer in enumerate(
        profile.name for profile, count in CUSTOMERS for _ in range(count)
    )
)


@router.get("/tickets")
async def list_tickets(cursor: Annotated[int, Query(ge=0)] = 0) -> TicketPage:
    end = cursor + PAGE_SIZE
    return TicketPage(
        items=list(TICKETS[cursor:end]),
        next_cursor=end if end < len(TICKETS) else None,
    )
