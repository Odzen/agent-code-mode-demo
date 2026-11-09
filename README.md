# Agent Run Lab

Two chats answer the same question over 18 synthetic support tickets. Tool Calling sends each page back to the model. Code Mode runs a Python loop in Monty and sends back the matching counts. Both agents use the same paginated FastAPI endpoint and stream their tool calls, response, and metrics as server-sent events. The frontend uses the shadcn/ui AI Elements chat pattern.

## Run locally

Requires Python 3.14, uv, and Node.js 24 or later. Copy `.env.example` to `.env` and set `HF_TOKEN` to a Hugging Face token with Inference Providers permission.

```sh
uv sync --locked
npm --prefix frontend ci
npm --prefix frontend run build
uv run --env-file .env fastapi dev
```

Open `http://127.0.0.1:8000/?example=enterprise-slas` to start with the Enterprise SLAs question. Run both agents and expand **Execution trace** to see each tool's output and how Monty ran the generated Python.

Both agents use `deepseek-ai/DeepSeek-V4.1-Flash-1017:novita` through Hugging Face Inference Providers. The server supplies the token for both runs.

## Verify

```sh
uv run python -m unittest discover -s tests
npm --prefix frontend run lint
npm --prefix frontend run build
```

## Deploy to FastAPI Cloud

Add `HF_TOKEN` as a secret in FastAPI Cloud before deploying. Build the frontend before each deployment. FastAPI serves `frontend/dist` and the API from the same app. The deployment include rule uploads the compiled frontend.

```sh
npm --prefix frontend ci
npm --prefix frontend run build
uv run fastapi cloud login
uv run fastapi deploy
```
