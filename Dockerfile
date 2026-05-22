FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project

COPY src ./src
COPY server.py ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

# Download models
RUN mkdir -p /app/models && \
    wget -q -O /app/models/phonikud-1.0.int8.onnx \
        "https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx" && \
    wget -q -O /app/models/shaul.onnx \
        "https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/shaul.onnx" && \
    wget -q -O /app/models/michael.onnx \
        "https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/michael.onnx" && \
    wget -q -O /app/models/model.config.json \
        "https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.config.json"

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

EXPOSE 8880

CMD ["python", "server.py"]
