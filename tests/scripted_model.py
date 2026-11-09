import json
from collections import Counter
from collections.abc import AsyncIterator

from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

EXAMPLE_PROMPT = "Which customers have at least three overdue support tickets? Give me the counts."
EXAMPLE_CODE = """counts = {}
cursor = None

while True:
    page = await list_overdue_tickets(cursor=cursor)
    for ticket in page["items"]:
        customer = ticket["customer"]
        counts[customer] = counts.get(customer, 0) + 1
    cursor = page["next_cursor"]
    if cursor is None:
        break

[
    {"customer": customer, "overdue_tickets": count}
    for customer, count in sorted(counts.items())
    if count >= 3
]"""


async def scripted_stream(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:
    """Drive the fixed example with a deterministic model stream."""
    returns = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    tool = info.function_tools[0].name
    if tool == "run_code":
        if not returns:
            yield {0: DeltaToolCall(name=tool, json_args=json.dumps({"code": EXAMPLE_CODE}))}
            return
        content = returns[-1].content
        rows = content["result"] if isinstance(content, dict) else content
        counts = {row["customer"]: row["overdue_tickets"] for row in rows}
    else:
        cursor = returns[-1].model_response_object()["next_cursor"] if returns else 0
        if cursor is not None:
            yield {0: DeltaToolCall(name=tool, json_args=json.dumps({"cursor": cursor}))}
            return
        counts = Counter(ticket["customer"] for page in returns for ticket in page.model_response_object()["items"])

    yield "Customers with at least three overdue tickets:\n\n"
    for customer, count in sorted(counts.items()):
        if count >= 3:
            yield f"- {customer}: {count}\n"


def example_model() -> FunctionModel:
    return FunctionModel(stream_function=scripted_stream, model_name="scripted-example")
