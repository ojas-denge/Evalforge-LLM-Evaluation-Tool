FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY evaluation_baselines ./evaluation_baselines

# Install CPU-only PyTorch first.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Install EvalForge dependencies.
RUN pip install --no-cache-dir .

# Download the embedding model at image-build time
# into a deterministic local directory.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-small-en-v1.5', local_dir='/opt/models/bge-small-en-v1.5')"

# Create a dedicated non-root runtime user.
RUN useradd --create-home --uid 10001 evalforge \
    && mkdir -p /app/.evalforge_checkpoints \
    && chown -R evalforge:evalforge /app/.evalforge_checkpoints

# Use the baked model at runtime.
ENV EMBEDDING_MODEL=/opt/models/bge-small-en-v1.5

# Run the application as a non-root user.
USER evalforge

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
