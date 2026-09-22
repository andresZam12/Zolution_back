# =============================================================================
# Zolution Backend — Dockerfile
# Multi-stage build: builder stage installs deps, runtime stage is lean.
# TODO (Phase 2): Implement full multi-stage build when app code is in place.
# =============================================================================

# Stage 1 — dependency builder
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ----------------------------------------------------------------------------
# Stage 2 — runtime image
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Install only runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy application source
COPY . .

# Run as non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser
USER appuser

EXPOSE 8000

# Development: uvicorn with hot-reload
# Production: override CMD with gunicorn + uvicorn workers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
