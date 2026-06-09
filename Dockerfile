# Build frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Build backend
FROM python:3.11-slim
WORKDIR /app/backend

# Install system dependencies if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package installation
RUN pip install uv

# Copy backend code
COPY backend/ ./

# Install python dependencies using uv
# Assuming there is a requirements.txt, or we can use pyproject.toml
# Let's check if there's a requirements.txt or pyproject.toml
RUN uv pip install --system -r requirements.txt || uv pip install --system -e .

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