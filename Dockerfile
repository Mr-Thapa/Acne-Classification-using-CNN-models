FROM python:3.12-slim

WORKDIR /app
COPY requirements_deploy.txt .

RUN pip install --no-cache-dir -r requirements_deploy.txt

COPY app/ ./app/
COPY src/__init__.py ./src/__init__.py
COPY src/config.py ./src/config.py
COPY models/final/vgg_finetuned_trained.keras ./models/final/vgg_finetuned_trained.keras
COPY frontend/ ./frontend/

EXPOSE 8000

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]