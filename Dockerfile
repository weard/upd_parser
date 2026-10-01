FROM python:3.11-slim

# Tesseract + русский языковой пакет (единственные системные зависимости)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-rus \
        tesseract-ocr-eng \
        tesseract-ocr-osd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

# -u: вывод логов сразу в docker logs, без буферизации
CMD ["python", "-u", "app.py"]
