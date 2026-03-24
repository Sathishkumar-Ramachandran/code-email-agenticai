FROM python:3.13-slim

WORKDIR /app

# Install system dependencies for Playwright and trafilatura
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (Chromium only)
RUN playwright install --with-deps chromium

# Copy application code
COPY config.py .
COPY main.py .
COPY app.py .
COPY agents/ agents/

# Streamlit config: disable telemetry, set server options
RUN mkdir -p /root/.streamlit
RUN printf '[server]\nheadless = true\naddress = "0.0.0.0"\nenableCORS = false\nenableXsrfProtection = false\n\n[browser]\ngatherUsageStats = false\n' > /root/.streamlit/config.toml

# Cloud Run injects PORT env var (default 8080)
ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/_stcore/health || exit 1

# Use shell form so $PORT is expanded at runtime
CMD streamlit run app.py --server.port=${PORT}
