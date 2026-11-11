FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/home/user HF_HOME=/home/user/.cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libsndfile1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir -r /app/requirements.txt && python -m pip check
COPY app.py /app/app.py
RUN python -m py_compile /app/app.py
RUN useradd -m -u 1000 user && mkdir -p /tmp/telugu_voice_v2 && chown -R user:user /app /tmp/telugu_voice_v2
USER user
EXPOSE 7860
CMD ["python", "app.py"]
