FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ingest_koniq.py .
COPY upload_images.py .
COPY expand_synthetic.py .

CMD ["python", "ingest_koniq.py"]
