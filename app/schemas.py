from typing import Literal

from pydantic import BaseModel, Field

SiNo = Literal["Yes", "No"]
Servicio = Literal["Yes", "No", "No internet service"]


class Cliente(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1] = Field(..., description="1 si el cliente tiene 65 años o más")
    Partner: SiNo = Field(..., description="Tiene pareja")
    Dependents: SiNo = Field(..., description="Tiene personas a cargo")
    tenure: int = Field(..., ge=0, le=100, description="Meses que lleva como cliente")
    PhoneService: SiNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Servicio
    OnlineBackup: Servicio
    DeviceProtection: Servicio
    TechSupport: Servicio
    StreamingTV: Servicio
    StreamingMovies: Servicio
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: SiNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(..., gt=0, description="Cargo mensual en USD")
    TotalCharges: float | None = Field(None, ge=0, description="Total cobrado hasta hoy en USD (null si es cliente nuevo)")


class PeticionPrediccion(BaseModel):
    clientes: list[Cliente] = Field(..., min_length=1, max_length=1000)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "clientes": [
                    {"gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
                     "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
                     "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                     "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
                     "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                     "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 190.0},
                    {"gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
                     "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
                     "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
                     "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
                     "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
                     "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 75.0,
                     "TotalCharges": 4500.0},
                ]
            }]
        }
    }


class Prediccion(BaseModel):
    prob_churn: float
    prediccion: Literal["se va", "se queda"]


class RespuestaPrediccion(BaseModel):
    modelo: str
    version: str
    umbral: float
    predicciones: list[Prediccion]