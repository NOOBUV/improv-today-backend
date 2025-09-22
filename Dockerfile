FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Default to production command; override with --build-arg ENVIRONMENT=development for dev
ARG ENVIRONMENT=production
ENV ENVIRONMENT=$ENVIRONMENT

# Gunicorn config: 2-4 workers typical for small instances; adjust via env
ENV WEB_CONCURRENCY=2
ENV WORKER_CLASS=uvicorn.workers.UvicornWorker

# Default command - can be overridden in docker-compose
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]