# StatVerify AI — Production Dockerfile
FROM python:3.12-slim

# Install system dependencies (Tesseract OCR, Poppler for PDF image extraction)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir streamlit pandas numpy scipy statsmodels reportlab pypdf pdfplumber Pillow google-generativeai pydantic pydantic-settings python-dotenv loguru ruff pytest

# Copy application source code
COPY src/ ./src/
COPY assets/ ./assets/
COPY data/ ./data/
COPY app.py .
COPY .env.example .env

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to monitor server responsiveness
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit server
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
