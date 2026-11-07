from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS, agregar_variables


def construir_pipeline() -> Pipeline:
    preprocesamiento = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), COLUMNAS_NUMERICAS),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), COLUMNAS_CATEGORICAS),
    ])

    return Pipeline([
        ("features", FunctionTransformer(agregar_variables)),
        ("preprocesamiento", preprocesamiento),
        ("modelo", LogisticRegression(max_iter=2000)),
    ])


GRID = [
    {"modelo": [LogisticRegression(max_iter=2000)],
     "modelo__C": [0.1, 1, 10]},
    {"modelo": [HistGradientBoostingClassifier(early_stopping=True, random_state=42)],
     "modelo__learning_rate": [0.05, 0.1],
     "modelo__max_leaf_nodes": [7, 15]},
]