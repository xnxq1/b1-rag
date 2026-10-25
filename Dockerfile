FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    FASTEMBED_CACHE_PATH=/opt/fastembed

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

RUN /opt/venv/bin/python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25', language='russian')"


FROM python:3.12-slim AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    HF_HUB_OFFLINE=1 \
    STORAGE_DIR=/data/uploads

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data/uploads \
    && chown app:app /data/uploads

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/fastembed /opt/fastembed

WORKDIR /app
COPY rag ./rag
COPY manage.py ./

USER app

EXPOSE 8000
CMD ["python", "manage.py", "start-api"]
