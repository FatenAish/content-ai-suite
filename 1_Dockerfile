FROM python:3.10-slim

# Set working directory
WORKDIR /app

# ----------------------------------------------------
# Install system dependencies required by FAISS + ST
# ----------------------------------------------------
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libstdc++6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------
# Install Python requirements
# ----------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------
# Copy application code
# ----------------------------------------------------
COPY . .

# ----------------------------------------------------
# Ensure data folder exists inside the container
# ----------------------------------------------------
RUN mkdir -p /app/data

# ----------------------------------------------------
# Expose port for Streamlit
# ----------------------------------------------------
EXPOSE 8080

# ----------------------------------------------------
# Run Streamlit on Cloud Run
# ----------------------------------------------------
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
