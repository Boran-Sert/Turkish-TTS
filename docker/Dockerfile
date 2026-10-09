# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Bağımlılıklar ayrı katmanda: kaynak değişince yeniden kurulmaz.
COPY pyproject.toml README.md LICENSE ./
COPY src/turkish_tts/__init__.py src/turkish_tts/__init__.py
RUN uv pip install --system torch==2.5.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && uv pip install --system .

COPY src/ src/
COPY tts_config.json ./
RUN uv pip install --system --no-deps .

EXPOSE 8000
VOLUME ["/cache/huggingface"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["turkish-tts-server"]
