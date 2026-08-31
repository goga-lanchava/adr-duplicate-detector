FROM python:3.11-slim

WORKDIR /app

# numpy / pandas / scipy / streamlit at the pinned versions all ship manylinux
# wheels for CPython 3.11, so no compiler toolchain is required.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app.py .

# Drop root for the running service.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8501

# python:3.11-slim ships no curl; probe the health endpoint with the stdlib
# instead (urlopen raises on any non-2xx response -> non-zero exit).
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD \
    python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", "--server.address=0.0.0.0", \
    "--server.headless=true", "--browser.gatherUsageStats=false"]
