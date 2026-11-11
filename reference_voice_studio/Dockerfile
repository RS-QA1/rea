FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends \
  git ffmpeg libsndfile1 ca-certificates \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip check
COPY app.py /app/app.py
RUN python -m py_compile /app/app.py
RUN useradd -m -u 1000 appuser && mkdir -p /tmp/telugu_reference_studio && \
    chown -R appuser:appuser /app /tmp/telugu_reference_studio
USER appuser
EXPOSE 7860
CMD ["python", "app.py"]
