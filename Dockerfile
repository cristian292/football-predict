FROM docker.io/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown appuser:appuser /data

COPY --chown=appuser:appuser matches.csv ./matches.csv

USER appuser

EXPOSE 8080

CMD ["python", "-m", "football_predictor", "--host", "0.0.0.0", "--port", "8080", "--data", "/app/matches.csv", "--work-dir", "/data"]
