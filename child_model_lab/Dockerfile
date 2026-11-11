FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860 HOME=/home/user HF_HOME=/home/user/.cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libsndfile1 espeak-ng ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt && python -m pip check
COPY app.py ./
RUN python -m py_compile app.py && useradd -m -u 1000 user && mkdir -p /tmp/child_voice_lab /home/user/.cache/huggingface && chown -R user:user /app /tmp/child_voice_lab /home/user/.cache
USER user
EXPOSE 7860
CMD ["python","app.py"]
