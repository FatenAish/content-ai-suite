FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by FAISS + sentence-transformers
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ENTIRE project to the container (including data folder)
COPY . .

# DEBUG: Show /app content at build time
RUN echo "---- DEBUG: Listing /app ----" && ls -R /app

# Expose port for Streamlit
EXPOSE 8080

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
