# Nectar offer API. The model bundle is NOT baked in: it is mounted at /app/artifacts, so a new
# weekly bundle ships without rebuilding the image.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NECTAR_ARTIFACTS_DIR=/app/artifacts

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY docker/requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 app && mkdir -p /app/artifacts && chown app /app/artifacts
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "nectar.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
