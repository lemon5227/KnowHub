FROM node:22-alpine AS ui
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev
COPY knowhub/ knowhub/
COPY examples/ examples/
COPY scripts/ scripts/
COPY --from=ui /ui/dist frontend/dist
RUN useradd --create-home --uid 10001 knowhub && chown -R knowhub:knowhub /app
USER knowhub
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8077
CMD ["uvicorn", "knowhub.api:build_app", "--factory", "--host", "0.0.0.0", "--port", "8077"]
