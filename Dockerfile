FROM python:3.10-slim

# Prevent Python from writing pyc files & buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Cloud Run uses $PORT, not 8080
ENV PORT=8080

EXPOSE 8080

# ENTRYPOINT — the only correct one for Cloud Run + Streamlit
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false
