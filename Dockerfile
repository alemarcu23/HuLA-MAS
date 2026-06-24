# Dockerfile for Search & Rescue Multi-Player Game

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# build-essential, python3-dev, libc-dev are needed for compiling gevent from source
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libc-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
# Port 3000: Flask visualization server
# Port 3001: MATRX API
EXPOSE 3000 3001

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Default command to run the game
# Users can override with custom parameters
CMD ["python", "main.py", \
     "--task-type", "official", \
     "--condition", "normal", \
     "--session-id", "default_session"]
