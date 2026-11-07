import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.utils import estimator_html_repr

from src.data import cargar_telco
from src.features import COLUMNAS_ENTRADA
from src.pipeline import GRID, construir_pipeline

SEED = 42
DIR_MODELOS = Path("models")
RUTA_PIPELINE = DIR_MODELOS / "pipeline_churn.joblib"
RUTA_METADATA = DIR_MODELOS / "metadata.json"
RUTA_DIAGRAMA = DIR_MODELOS / "pipeline.html"


def main() -> None:
    datos = cargar_telco()
    X, y = datos[COLUMNAS_ENTRADA], datos["churn"]
    print(f"{len(datos)} clientes | {y.mean():.1%} se dio de baja")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    busqueda = GridSearchCV(
        construir_pipeline(), GRID, scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1,
    )
    busqueda.fit(X_tr, y_tr)
    pipeline = busqueda.best_estimator_

    tabla = pd.DataFrame(busqueda.cv_results_)
    tabla["modelo"] = tabla["param_modelo"].map(lambda m: type(m).__name__)
    columnas = ["modelo"] + [c for c in tabla if c.startswith("param_modelo__")] + ["mean_test_score"]
    print(tabla[columnas].sort_values("mean_test_score", ascending=False).to_string(index=False))

    params = {k.replace("modelo__", ""): v for k, v in busqueda.best_params_.items() if k != "modelo"}
    params["modelo"] = type(pipeline.named_steps["modelo"]).__name__

    proba = pipeline.predict_proba(X_te)[:, 1]
    pred = pipeline.predict(X_te)
    metricas = {
        "auc_cv": round(float(busqueda.best_score_), 4),
        "auc_test": round(float(roc_auc_score(y_te, proba)), 4),
        "accuracy_test": round(float(accuracy_score(y_te, pred)), 4),
        "recall_test": round(float(recall_score(y_te, pred)), 4),
        "f1_test": round(float(f1_score(y_te, pred)), 4),
    }

    DIR_MODELOS.mkdir(exist_ok=True)
    joblib.dump(pipeline, RUTA_PIPELINE)
    metadata = {
        "modelo": "pipeline_churn",
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "entrenado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pasos": [nombre for nombre, _ in pipeline.steps],
        "columnas_entrada": COLUMNAS_ENTRADA,
        "mejores_hiperparametros": params,
        "metricas": metricas,
        "versiones": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "pandas": pd.__version__, "numpy": np.__version__},
    }
    RUTA_METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    RUTA_DIAGRAMA.write_text(estimator_html_repr(pipeline), encoding="utf-8")

    print(pipeline)
    print(f"\nModelo elegido: {params}")
    print(f"Métricas: {metricas}")
    print(f"Pipeline guardado en {RUTA_PIPELINE} ({RUTA_PIPELINE.stat().st_size / 1024:.1f} KB)")

    if os.getenv("MLFLOW_TRACKING_URI"):
        registrar_en_mlflow(params, metricas)


def registrar_en_mlflow(params, metricas) -> None:
    import mlflow

    mlflow.set_experiment("taller3-churn")
    with mlflow.start_run(run_name=f"pipeline-{params['modelo']}") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metricas)
        mlflow.set_tags({"curso": "MLOps EIA", "taller": "3"})
        mlflow.log_artifacts(str(DIR_MODELOS), artifact_path="pipeline")
        for archivo in Path("src").glob("*.py"):
            mlflow.log_artifact(str(archivo), artifact_path="pipeline/src")
        print(f"Run registrado en MLflow: {run.info.run_id}")
        print(f"Artefactos en: {run.info.artifact_uri}")


if __name__ == "__main__":
    main()