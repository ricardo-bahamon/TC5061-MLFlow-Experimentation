"""Módulo de construcción y entrenamiento de modelos scikit-learn."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_logistic_regression_pipeline(
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    penalty: str = "l2",
    random_state: int = 42,
) -> Pipeline:
    """Crea un Pipeline de scikit-learn estandarizado para regresión logística multiclase."""
    model_params = {
        "C": C,
        "solver": solver,
        "max_iter": max_iter,
        "random_state": random_state,
    }
    if penalty == "l1":
        if solver not in ["liblinear", "saga"]:
            raise ValueError(f"El solver '{solver}' no soporta penalización L1.")
        model_params["penalty"] = "l1"
    else:
        model_params["penalty"] = "l2"

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(**model_params)),
    ])
