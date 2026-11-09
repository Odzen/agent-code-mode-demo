# Agent Run Lab

Two chats answer the same question over 18 synthetic support tickets. Tool Calling sends each page back to the model. Code Mode runs a Python loop in Monty and sends back the matching counts. Both agents use the same paginated FastAPI endpoint and stream their tool calls, response, and metrics as server-sent events. The frontend uses the shadcn/ui AI Elements chat pattern.

## Run locally

Requires Python 3.14, uv, and Node.js 24 or later.

```sh
uv sync --locked
npm --prefix frontend ci
npm --prefix frontend run build
uv run fastapi dev
```

Open `http://127.0.0.1:8000`, enter your DeepSeek API key, and click **Run both**. Flash is the default, with V4 Pro also available. Expand **Execution trace** to see the API calls and the Python passed to Monty.

Both agents use your key for the selected model. The browser holds the key in memory and sends it in the authorization header for each run.

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
