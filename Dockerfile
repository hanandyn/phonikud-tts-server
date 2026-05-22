FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y wget libsndfile1 curl git && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY server.py ./

RUN mkdir -p /app/models

# Install deps + Renikud (better G2P, ~20MB)
RUN uv sync --frozen && uv pip install flask renikud-onnx

# Download models (Renikud 20MB replaces Phonikud 294MB)
RUN wget -q -O /app/models/renikud.onnx \
        "https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx" && \
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
