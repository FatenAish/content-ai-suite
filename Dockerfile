FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# 🔥 EXPLICITLY copy data folder (fixes your issue)
COPY ./data /app/data

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
