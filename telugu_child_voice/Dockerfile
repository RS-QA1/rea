FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.8.0+cpu torchaudio==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir omnivoice==0.2.1 'gradio>=5,<7' soundfile && pip check
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY app.py .
USER appuser
EXPOSE 7860
CMD ["python", "app.py"]
