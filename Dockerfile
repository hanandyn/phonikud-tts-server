FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y wget libsndfile1 curl git && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY server.py ./

RUN mkdir -p /app/models

RUN uv sync --frozen

RUN wget -q -O /app/models/phonikud-1.0.int8.onnx \
        "https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx" && \
    wget -q -O /app/models/shaul.onnx \
        "https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/shaul.onnx" && \
    wget -q -O /app/models/michael.onnx \
        "https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/michael.onnx" && \
    wget -q -O /app/models/model.config.json \
        "https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.config.json"

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8880

CMD ["python", "server.py"]
