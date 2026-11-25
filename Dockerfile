FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency and application files
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY .env.example ./

# Create venv and install dependencies
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"
RUN uv venv /venv && \
    uv pip install --python /venv/bin/python --no-cache .

RUN mkdir -p /app/results

# Expose ports
EXPOSE 8000 8501

# Default command: run API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
