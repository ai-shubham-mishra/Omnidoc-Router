FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create tmp directory for file uploads
RUN mkdir -p /app/tmp

# Copy application code
COPY app.py .
COPY core/ ./core/
COPY handlers/ ./handlers/
COPY models/ ./models/
COPY auth/ ./auth/
COPY utils/ ./utils/

EXPOSE 8500

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8500", "--workers", "3", "--timeout-keep-alive", "600"]
