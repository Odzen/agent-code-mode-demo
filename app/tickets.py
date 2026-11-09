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


CUSTOMERS = (
    ("Aster Labs", 6),
    ("Ember Works", 5),
    ("Juniper Studio", 3),
    ("Signal House", 2),
    ("Cedar Systems", 1),
    ("Northstar Print", 1),
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
        customer for customer, count in CUSTOMERS for _ in range(count)
    )
)


@router.get("/tickets")
async def list_tickets(cursor: Annotated[int, Query(ge=0)] = 0) -> TicketPage:
    end = cursor + PAGE_SIZE
    return TicketPage(
        items=list(TICKETS[cursor:end]),
        next_cursor=end if end < len(TICKETS) else None,
    )
