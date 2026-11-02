FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Create venv and install dependencies
RUN uv venv /venv && \
    /venv/bin/pip install --no-deps -e .

# Copy application code
COPY src/ ./src/
COPY .env.example ./

# Expose ports
EXPOSE 8000 8501

# Default command: run API + UI
CMD ["/venv/bin/uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]