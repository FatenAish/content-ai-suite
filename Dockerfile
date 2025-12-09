FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by FAISS + sentence-transformers
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full application
COPY . .

# IMPORTANT: copy data folder into the container
COPY data /app/data

# Expose port for Streamlit
EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
