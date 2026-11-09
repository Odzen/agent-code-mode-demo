import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from scripted_model import EXAMPLE_PROMPT, example_model


def read_events(text: str) -> list[tuple[str, object]]:
    return [
        (event.removeprefix("event: "), json.loads(data.removeprefix("data: ")))
        for event, data in (
            block.splitlines() for block in text.strip().split("\n\n") if block.startswith("event:")
        )
    ]


class ComparisonTests(unittest.TestCase):
    def test_agent_runs_require_an_api_key(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/run", json={"approach": "tool-calling", "prompt": EXAMPLE_PROMPT}
            )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")

    @patch("app.main.OpenAIChatModel", side_effect=lambda *_args, **_kwargs: example_model())
    def test_same_answer_and_pages_with_less_model_coordination(self, _model):
        with TestClient(app) as client:
            runs = {}
            for approach in ("tool-calling", "code-mode"):
                response = client.post(
                    "/api/run",
                    json={"approach": approach, "prompt": EXAMPLE_PROMPT},
                    headers={"Authorization": "Bearer test-key"},
                )
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/event-stream", response.headers["content-type"])
                runs[approach] = read_events(response.text)

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
        self.assertEqual([data["cursor"] for event, data in code if event == "page"], [None, 6, 12])
