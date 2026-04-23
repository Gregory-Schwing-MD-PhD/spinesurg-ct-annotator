# SpineSurg CT Annotator — Hugging Face Docker Space image.
#
# Base: projectmonai/monailabel contains MONAI Label + OHIF bundled assets.
# We layer FastAPI + huggingface_hub + authlib on top and replace the
# container entrypoint with our own reverse-proxying FastAPI server.

FROM projectmonai/monailabel:latest

USER root

# --- OS packages ---------------------------------------------------------- #
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# --- Workspace ------------------------------------------------------------ #
# HF Spaces runs as a non-root user (uid 1000). Make workspace writable.
RUN mkdir -p /workspace/raw_data /workspace/labels \
 && chmod -R 777 /workspace

ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WORKSPACE=/workspace \
    HF_HOME=/workspace/.hf_cache

RUN mkdir -p $HOME $HF_HOME && chmod -R 777 $HOME $HF_HOME

# --- Python deps ---------------------------------------------------------- #
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# --- App ------------------------------------------------------------------ #
COPY app.py config.py audit.py sync_manager.py ./
COPY assignments.json ./
COPY README.md ./

# --- Runtime -------------------------------------------------------------- #
EXPOSE 7860

# FastAPI is the sole public listener; it spawns MONAI Label on 127.0.0.1:8000
# via the lifespan hook in app.py.
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*", \
     "--no-access-log"]
