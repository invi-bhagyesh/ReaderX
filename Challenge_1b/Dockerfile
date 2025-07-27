FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libcrypt1 \
    && ln -s /usr/lib/aarch64-linux-gnu/libcrypt.so.1 /usr/lib/aarch64-linux-gnu/libcrypt.so.2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy app
COPY app.py .

# Create input/output dirs
RUN mkdir -p /app/input /app/output

ENTRYPOINT ["python", "app.py"]
