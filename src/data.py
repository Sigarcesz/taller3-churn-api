"""Dataset Telco Customer Churn (IBM).

7.043 clientes de una empresa de telecomunicaciones: datos demográficos,
servicios contratados, tipo de contrato y cargos. La etiqueta Churn indica si
el cliente se dio de baja en el último mes.
"""
from pathlib import Path

import pandas as pd

URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
       "master/data/Telco-Customer-Churn.csv")
RUTA_LOCAL = Path("data/telco_churn.csv")


def cargar_telco() -> pd.DataFrame:
    """Descarga el CSV la primera vez y lo guarda en data/ para no volver a bajarlo."""
    if not RUTA_LOCAL.exists():
        RUTA_LOCAL.parent.mkdir(exist_ok=True)
        pd.read_csv(URL).to_csv(RUTA_LOCAL, index=False)

    df = pd.read_csv(RUTA_LOCAL)
    # Faltante escondido: TotalCharges viene como texto y los clientes nuevos
    # (tenure = 0) tienen un espacio en blanco en vez de un número.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    # customerID es un identificador: no aporta nada al modelo
    return df.drop(columns=["customerID", "Churn"])