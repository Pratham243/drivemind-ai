# DriveMind AI — FastAPI backend image
FROM python:3.11-slim AS base

WORKDIR /app

# System deps needed by scikit-learn/matplotlib at build time are minimal;
# torch/transformers are intentionally NOT installed in the default image
# (requirements-ml.txt) to keep the container small and fast to build —
# the API runs on the TF-IDF baseline by default (NLU_MODEL=baseline).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY data/ data/
COPY models/ models/
COPY app/ app/

ENV PYTHONPATH=/app
ENV NLU_MODEL=baseline

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
