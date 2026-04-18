FROM python:3.11-slim

# Install system dependencies (ffmpeg for pydub)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first to avoid downloading GPU packages
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create output directory for digests
RUN mkdir -p out

# Set proper permissions
RUN chmod +x scripts/fetch_and_publish_digest.py

# Run the script
CMD ["python", "scripts/fetch_and_publish_digest.py"]
