# -----------------------------------------------------
# Base Python Image
# -----------------------------------------------------
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL project files (including /data folder)
COPY . .

# Ensure /app/data exists
RUN mkdir -p /app/data

# Streamlit configuration for Cloud Run
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
