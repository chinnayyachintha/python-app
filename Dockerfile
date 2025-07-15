# Use a stable and lightweight Python image
FROM python:3.10-slim

# Environment settings for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create and set working directory
WORKDIR /app

# Optional: Install system-level dependencies (e.g., for numpy, pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements file first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy rest of the application
COPY . .

# Expose port if it's a web server like Flask or FastAPI
EXPOSE 5000

# Command to run your app — change to your actual entry point
CMD ["python", "app.py"]
