FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y wget libsndfile1 curl git && rm -rf /var/lib/apt/lists/*
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY server.py ./
RUN mkdir -p /app/models
RUN uv sync --frozen && uv pip install flask renikud-onnx
RUN wget -q -O /app/models/renikud.onnx \
        "https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx" && \
    wget -q -O /app/models/shaul.onnx \
        "https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/shaul.onnx" && \
    wget -q -O /app/models/michael.onnx \
        "https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/michael.onnx" && \
    wget -q -O /app/models/model.config.json \
        "https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.config.json" && \
    wget -q -O /app/models/libritts_hebrew.onnx \
        "https://github.com/thewh1teagle/style-onnx/releases/download/model-files-v1.0/libritts_hebrew.onnx" && \
    wget -q -O /app/models/636_female_style.npy \
        "https://github.com/thewh1teagle/style-onnx/releases/download/model-files-v1.0/636_female_style.npy" && \
    wget -q -O /app/models/female1_style.npy \
        "https://github.com/thewh1teagle/style-onnx/releases/download/model-files-v1.0/style_female1.npy"
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH="/app" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
EXPOSE 8880
CMD ["python", "server.py"]
