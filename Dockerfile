FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY data/processed/ data/processed/
EXPOSE 8000
# Models are trained inside container with: python -m src.models.trainer
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
