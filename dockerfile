FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libssl-dev \
    libffi-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p input output models/tinyllama

# Copy the application code
COPY solution.py .

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_VERBOSITY=error
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

# Expose volume mounts
VOLUME ["/app/input", "/app/output", "/app/models"]

# Run the application
CMD ["python", "solution.py"]
