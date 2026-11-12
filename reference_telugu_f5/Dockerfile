FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/home/user HF_HOME=/home/user/.cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git espeak-ng libsndfile1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt && python -m pip check
COPY app.py ./
RUN python -m py_compile app.py && useradd -m -u 1000 user && mkdir -p /tmp/telugu-reference-child && chown -R user:user /app /tmp/telugu-reference-child
USER user
EXPOSE 7860
CMD ["python","app.py"]
