import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from app.agents import DATA_TOOL_NAMES
from app.main import HF_MODEL, HF_ROUTER, app
from scripted_model import EXAMPLE_PROMPT, example_model


def read_events(text: str) -> list[tuple[str, object]]:
    return [
        (event.removeprefix("event: "), json.loads(data.removeprefix("data: ")))
        for event, data in (
            block.splitlines() for block in text.strip().split("\n\n") if block.startswith("event:")
        )
    ]


class ComparisonTests(unittest.TestCase):
    @patch.dict("os.environ", {"HF_TOKEN": ""})
    def test_agent_runs_require_a_server_token(self):
        with TestClient(app) as client:
            self.assertEqual(client.get("/api/config").json(), {"model": HF_MODEL, "ready": False, "tools": DATA_TOOL_NAMES})
            response = client.post(
                "/api/run", json={"approach": "tool-calling", "prompt": EXAMPLE_PROMPT}
            )
        self.assertEqual(response.status_code, 503)

    @patch.dict("os.environ", {"HF_TOKEN": "test-token"})
    @patch("app.main.OpenAIChatModel", side_effect=lambda *_args, **_kwargs: example_model())
    def test_same_answer_and_pages_with_less_model_coordination(self, model):
        with TestClient(app) as client:
            self.assertEqual(client.get("/api/config").json(), {"model": HF_MODEL, "ready": True, "tools": DATA_TOOL_NAMES})
            runs = {}
            for approach in ("tool-calling", "code-mode"):
                response = client.post(
                    "/api/run",
                    json={"approach": approach, "prompt": EXAMPLE_PROMPT},
                )
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/event-stream", response.headers["content-type"])
                runs[approach] = read_events(response.text)

        self.assertEqual([call.args[0] for call in model.call_args_list], [HF_MODEL, HF_MODEL])
        self.assertEqual(
            [str(call.kwargs["provider"].base_url).rstrip("/") for call in model.call_args_list],
            [HF_ROUTER, HF_ROUTER],
        )
        direct, code = runs.values()
        self.assertEqual(direct[-1][0], "done")
        self.assertEqual(code[-1][0], "done")
        direct_result, code_result = direct[-1][1], code[-1][1]
        self.assertEqual(direct_result["answer"], code_result["answer"])
        self.assertIn("Juniper Studio: 3", code_result["answer"])
        self.assertEqual(direct_result["pages_read"], 3)
        self.assertEqual(code_result["pages_read"], 3)
        self.assertEqual(direct_result["model_requests"], 4)
        self.assertEqual(code_result["model_requests"], 2)
        self.assertLess(code_result["tool_output_bytes"], direct_result["tool_output_bytes"])
        self.assertEqual([data["name"] for event, data in code if event == "tool"], ["run_code"])
        code_call = next(data for event, data in code if event == "tool")
        code_pages = [data for event, data in code if event == "page"]
        self.assertEqual([page["cursor"] for page in code_pages], [None, 6, 12])
        self.assertEqual({page["parent_id"] for page in code_pages}, {code_call["call_id"]})
        self.assertEqual(len({page["call_id"] for page in code_pages}), 3)
        direct_outputs = [data["output"] for event, data in direct if event == "tool_result"]
        self.assertEqual(direct_outputs, [page["output"] for page in code_pages])
        self.assertEqual(direct_outputs[0]["next_cursor"], 6)
        code_output = next(data for event, data in code if event == "tool_result")
        self.assertEqual(code_output["call_id"], code_call["call_id"])
        self.assertEqual(code_output["outcome"], "success")
        self.assertIn({"customer": "Juniper Studio", "overdue_tickets": 3}, code_output["output"])
        self.assertEqual(
            [data["call_id"] for event, data in direct if event == "page"],
            [data["call_id"] for event, data in direct if event == "tool"],
        )

    @patch.dict("os.environ", {"HF_TOKEN": "test-token"})
    def test_code_trace_distinguishes_retry_from_success_without_nested_calls(self):
        async def stream(messages, _info):
            returns = [
                part
                for message in messages if isinstance(message, ModelRequest)
                for part in message.parts if isinstance(part, ToolReturnPart)
            ]
            if len(returns) < 2:
                code = "if" if not returns else "20 + 22"
                yield {0: DeltaToolCall(name="run_code", json_args=json.dumps({"code": code}))}
            else:
                yield "42"

        with patch("app.main.OpenAIChatModel", return_value=FunctionModel(stream_function=stream)):
            with TestClient(app) as client:
                response = client.post("/api/run", json={"approach": "code-mode", "prompt": "Compute 20 + 22."})
        events = read_events(response.text)
        self.assertEqual(events[-1][0], "done")
        outputs = [data for event, data in events if event == "tool_result"]
        self.assertEqual([output["outcome"] for output in outputs], ["retried", "success"])
        self.assertIn(outputs[0]["detail"], {"Syntax error in code", "Type error in code"})
        self.assertEqual(outputs[1]["output"], 42)
        self.assertFalse(any(event in {"page", "activity"} for event, _ in events))
