import numpy as np
import pandas as pd

SERVICIOS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
             "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

COLUMNAS_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges",
                      "cargo_promedio", "num_servicios"]
COLUMNAS_CATEGORICAS = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
                        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                        "Contract", "PaperlessBilling", "PaymentMethod"]

COLUMNAS_ENTRADA = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                    "MonthlyCharges", "TotalCharges"]


def agregar_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cargo_promedio"] = df["TotalCharges"] / np.maximum(df["tenure"], 1)
    df["num_servicios"] = (df[SERVICIOS] == "Yes").sum(axis=1)
    return df