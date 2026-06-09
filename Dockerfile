# Build frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app
COPY locales/ ./locales/
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Build backend
FROM python:3.11-slim
WORKDIR /app
COPY locales/ ./locales/
WORKDIR /app/backend

# Install system dependencies if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend code
COPY backend/ ./

# Install python dependencies using standard pip to avoid HF cache miss bugs
RUN pip install --no-cache-dir -r requirements.txt

# Copy built frontend static files
COPY --from=frontend-builder /app/frontend/dist /app/backend/app/static

# Setup environment variables for HF Spaces
ENV PORT=7860
ENV FLASK_PORT=7860
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "run.py"]