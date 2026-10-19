# Streamlit app image. Secrets are never baked in: pass them at run time (see README).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes don't invalidate this cached layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

# The package, then the app and its data.
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-deps .
COPY app.py .
COPY corpus/ corpus/

# Run as an unprivileged user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
