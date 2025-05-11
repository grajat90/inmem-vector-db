# Use Python 3.10 slim as the base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies and curl for health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default environment variables (can be overridden by .env file)
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    DATA_DIRECTORY=/app/data

# Create data directory if it doesn't exist
RUN mkdir -p ${DATA_DIRECTORY}

# Install python-dotenv to load .env file
RUN pip install --no-cache-dir python-dotenv

# Expose application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Command to run the application with .env file support
CMD ["sh", "-c", "python -m app.main"] 