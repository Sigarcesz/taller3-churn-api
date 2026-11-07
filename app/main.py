import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import PeticionPrediccion, Prediccion, RespuestaPrediccion
from src.features import COLUMNAS_ENTRADA

RUTA_MODELO = Path(os.getenv("MODEL_PATH", "models/pipeline_churn.joblib"))
RUTA_METADATA = RUTA_MODELO.with_name("metadata.json")
UMBRAL = float(os.getenv("UMBRAL", "0.5"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("churn-api")
estado: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    estado["pipeline"] = joblib.load(RUTA_MODELO)
    estado["metadata"] = json.loads(RUTA_METADATA.read_text(encoding="utf-8")) if RUTA_METADATA.exists() else {}
    log.info("Pipeline cargado desde %s (versión %s)", RUTA_MODELO, estado["metadata"].get("version", "?"))
    yield
    estado.clear()


app = FastAPI(
    title="API de predicción de churn",
    description="Taller 3 MLOps (Universidad EIA): predice si un cliente de telecomunicaciones se dará de baja (dataset Telco Customer Churn de IBM).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["info"])
def inicio():
    return {"servicio": "API de predicción de churn", "curso": "MLOps - Universidad EIA",
            "docs": "/docs", "salud": "/health"}


@app.get("/health", tags=["info"])
def health():
    return {"status": "ok", "modelo_cargado": "pipeline" in estado,
            "version_modelo": estado.get("metadata", {}).get("version")}


@app.get("/model/info", tags=["info"])
def model_info():
    return estado.get("metadata", {})


@app.post("/predict", response_model=RespuestaPrediccion, tags=["predicción"])
def predict(peticion: PeticionPrediccion):
    df = pd.DataFrame([s.model_dump() for s in peticion.clientes], columns=COLUMNAS_ENTRADA)
    df = df.fillna(value=np.nan)
    try:
        proba = estado["pipeline"].predict_proba(df)[:, 1]
    except Exception as e:
        log.exception("Error al predecir")
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

    log.info("Predicción para %d cliente(s)", len(df))
    meta = estado.get("metadata", {})
    return RespuestaPrediccion(
        modelo=meta.get("modelo", "pipeline_churn"),
        version=meta.get("version", "desconocida"),
        umbral=UMBRAL,
        predicciones=[Prediccion(prob_churn=round(float(p), 4),
                                 prediccion="se va" if p >= UMBRAL else "se queda") for p in proba],
    )