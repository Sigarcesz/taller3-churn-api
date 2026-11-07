from fastapi.testclient import TestClient

from app.main import app

CLIENTE = {"gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
           "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
           "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
           "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
           "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
           "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 190.0}


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["modelo_cargado"] is True


def test_predict_ok():
    with TestClient(app) as client:
        r = client.post("/predict", json={"clientes": [CLIENTE]})
        assert r.status_code == 200
        pred = r.json()["predicciones"][0]
        assert 0 <= pred["prob_churn"] <= 1
        assert pred["prediccion"] in {"se va", "se queda"}


def test_predict_cliente_nuevo_sin_total():
    with TestClient(app) as client:
        nuevo = {**CLIENTE, "tenure": 0, "TotalCharges": None}
        r = client.post("/predict", json={"clientes": [nuevo]})
        assert r.status_code == 200


def test_predict_dato_invalido():
    with TestClient(app) as client:
        r = client.post("/predict", json={"clientes": [{**CLIENTE, "Contract": "Three year"}]})
        assert r.status_code == 422