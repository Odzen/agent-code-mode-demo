# Agent Run Lab

Compare Tool Calling and Code Mode on the same question. Both agents use the same tools to read 18 synthetic tickets, look up customer profiles, or search public Pydantic AI issues. In Code Mode, the model writes Python that calls those tools inside Monty and returns a result to the model.

## Run locally

Requires Python 3.14, uv, and Node.js 24 or later. Copy `.env.example` to `.env` and set `HF_TOKEN` to a Hugging Face token with Inference Providers permission.

```sh
uv sync --locked
npm --prefix frontend ci
npm --prefix frontend run build
uv run --env-file .env fastapi dev
```

Open `http://127.0.0.1:8000/?example=enterprise-slas` and run both agents. Open **Execution trace** to inspect the tool outputs and generated Python.

Both agents use `deepseek-ai/DeepSeek-V4.1-Flash-1017:novita` through Hugging Face Inference Providers. The server supplies the token for both runs.

## Verify

```sh
uv run python -m unittest discover -s tests
npm --prefix frontend run lint
npm --prefix frontend run build
```

## Deploy to FastAPI Cloud

Add `HF_TOKEN` as a secret in FastAPI Cloud. Build the frontend before deploying so FastAPI can serve it from `frontend/dist`.

```sh
npm --prefix frontend ci
npm --prefix frontend run build
uv run fastapi cloud login
uv run fastapi deploy
```
