FROM python:3.11-slim

WORKDIR /app

# ffmpeg and libsndfile support audio decoding in model mode.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-models.txt ./
ARG INSTALL_MODELS=0
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_MODELS" = "1" ]; then \
        pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
        && pip install --no-cache-dir -r requirements-models.txt; \
    fi

COPY app/ ./app/

ENV PYTHONUNBUFFERED=1 \
    AI_LOAD_MODELS=0
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
