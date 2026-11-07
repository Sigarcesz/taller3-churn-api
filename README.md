# API de predicción de churn · Taller 3 MLOps (Universidad EIA)

API FastAPI que sirve un pipeline de scikit-learn (variables propias + preprocesamiento + modelo) entrenado con el dataset Telco Customer Churn de IBM. Predice si un cliente se dará de baja.

## Ejecutar en local (Python 3.11 o superior)

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements-dev.txt
    python train.py        # opcional: el .joblib ya viene en el repo
    pytest -v
    uvicorn app.main:app --reload        # http://127.0.0.1:8000/docs
    curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @ejemplo.json

## Desplegar en EC2 (Ubuntu)

    sudo apt update && sudo apt install -y python3-venv python3-pip git
    git clone https://github.com/Sigarcesz/taller3-churn-api.git && cd taller3-churn-api
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    sudo cp deploy/churn-api.service /etc/systemd/system/
    sudo systemctl daemon-reload && sudo systemctl enable --now churn-api

Security group: puerto 8000 abierto. Endpoints: GET /health, GET /model/info, POST /predict, docs en /docs.
