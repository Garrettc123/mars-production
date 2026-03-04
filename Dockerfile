FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi uvicorn anthropic
COPY mars_api.py .
CMD uvicorn mars_api:app --host 0.0.0.0 --port ${PORT:-8000}
