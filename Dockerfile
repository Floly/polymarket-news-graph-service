FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ python3-dev libffi-dev && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

RUN apt-get purge -y --auto-remove gcc g++ python3-dev libffi-dev

COPY app/ ./app/
COPY models/ ./models/
COPY logs/ ./logs/
COPY data/ ./data/

ENV PYTHONPATH=/app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]