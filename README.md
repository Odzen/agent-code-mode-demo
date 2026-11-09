# Agent Run Lab

Two chats answer the same question over 18 synthetic support tickets. Tool Calling sends each page back to the model. Code Mode runs a Python loop in Monty and sends back the matching counts. Both agents use the same paginated FastAPI endpoint and stream their tool calls, response, and metrics as server-sent events.

## Run locally

Requires Python 3.14, uv, and Node.js 24 or later.

```sh
uv sync --locked
npm --prefix frontend ci
npm --prefix frontend run build
uv run fastapi dev
```

Open `http://127.0.0.1:8000` and click **Run both**. Expand **Execution trace** to see the API calls and the Python passed to Monty.

**Example** uses a scripted model for the fixed question. The ticket endpoint and Monty still execute on each run. **Live** accepts a custom question and a DeepSeek key. Flash is the default, with V4 Pro also available. The browser keeps a pasted key in memory and sends it to the backend with each live request. For local use, you can also copy `.env.example` to `.env`, set `DEEPSEEK_API_KEY`, and start with `uv run --env-file .env fastapi dev`.

## Example result

| | Tool Calling | Code Mode |
| --- | ---: | ---: |
| Model requests | 4 | 2 |
| Ticket pages read | 3 | 3 |
| Tool output added to model history | 1,818 bytes | 144 bytes |

Both report Aster Labs (6), Ember Works (5), and Juniper Studio (3). The byte counts measure serialized tool returns once each. Live runs also show the provider's input token count, which includes instructions, tool schemas, and conversation history.

## Verify

```sh
uv run python -m unittest discover -s tests
npm --prefix frontend run lint
npm --prefix frontend run build
```

## Deploy to FastAPI Cloud

Build the frontend before each deployment. FastAPI serves `frontend/dist` and the API from the same app. The deployment include rule uploads the compiled frontend.

```sh
npm --prefix frontend ci
npm --prefix frontend run build
uv run fastapi cloud login
uv run fastapi deploy
```

For a public deployment, leave `DEEPSEEK_API_KEY` unset so live runs use the reader’s own key.
