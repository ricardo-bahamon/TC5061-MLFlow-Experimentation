"""
TC5061 - Operaciones de Aprendizaje Automático (MLOps)
Semana 4: Código de experimentación reutilizable instrumentado con MLFlow

Autor: Jesus Ricardo Bahamon Quimbaya (Matrícula: A01840023)
Descripción:
    Este script migra el pipeline de clasificación de Wine Quality a un script
    de Python modular, parametrizado mediante argparse y completamente
    instrumentado con MLflow Tracking.
"""

import argparse
import json
import os
import sys

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Carga los datos desde un archivo CSV detectando automáticamente el separador.

    Args:
        filepath: Ruta relativa o absoluta al archivo CSV.

    Returns:
        DataFrame con los datos limpios.
    """
    path = Path(filepath)
    if not path.exists():
        # Fallback a búsqueda alternativa
        candidate_paths = [
            path,
            Path("data/raw/winequality-red.csv"),
            Path("../data/raw/winequality-red.csv"),
            Path("TC5061_MLFlow_Experimentation/data/raw/winequality-red.csv"),
        ]
        found = False
        for cand in candidate_paths:
            if cand.exists():
                path = cand
                found = True
                break
        if not found:
            raise FileNotFoundError(f"No se encontró el archivo de datos en: {filepath}")

    # Detectar delimitador (el CSV UCI de vino suele usar punto y coma)
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        delimiter = ";" if ";" in first_line else ","

    df = pd.read_csv(path, sep=delimiter)
    # Estandarizar nombres de columnas a snake_case
    df.columns = [c.strip().lower().replace(" ", "_").replace('"', "") for c in df.columns]
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    return df


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "quality",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa variables predictoras y objetivo, y divide en train y test.

    Args:
        df: DataFrame original.
        target_col: Nombre de la columna objetivo.
        test_size: Proporción del conjunto de prueba.
        random_state: Semilla aleatoria para garantizar reproducibilidad.

    Returns:
        Tupla con (X_train, X_test, y_train, y_test).
    """
    if target_col not in df.columns:
        raise ValueError(f"Columna objetivo '{target_col}' no encontrada en el DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_pipeline(
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    penalty: str = "l2",
    random_state: int = 42,
) -> Pipeline:
    """Construye un pipeline de scikit-learn con StandardScaler y LogisticRegression.

    Args:
        C: Inverso de la fuerza de regularización.
        solver: Algoritmo de optimización ('lbfgs', 'saga', 'liblinear').
        max_iter: Número máximo de iteraciones.
        penalty: Tipo de norma de penalización ('l2', 'l1').
        random_state: Semilla aleatoria.

    Returns:
        Instancia de Pipeline configurada.
    """
    model_params = {
        "C": C,
        "solver": solver,
        "max_iter": max_iter,
        "random_state": random_state,
    }

    # Restricciones de penalización según el solver
    if penalty == "l1":
        if solver not in ["liblinear", "saga"]:
            raise ValueError(f"El solver '{solver}' no soporta penalización l1. Usa 'liblinear' o 'saga'.")
        model_params["penalty"] = "l1"
    else:
        model_params["penalty"] = "l2"

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(**model_params)),
    ])
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
) -> Tuple[Dict[str, float], np.ndarray, str, Dict]:
    """Evalúa el modelo calculando múltiples métricas, matriz de confusión y validación cruzada.

    Args:
        pipeline: Pipeline entrenado.
        X_train: Características de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        X_test: Características de prueba.
        y_test: Etiquetas de prueba.
        cv_folds: Número de pliegues para validación cruzada.

    Returns:
        Tupla con (métricas, matriz_confusión, reporte_texto, reporte_dict).
    """
    y_pred = pipeline.predict(X_test)

    # Cálculo de métricas sobre test set
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    precision_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    # Validación cruzada sobre el conjunto de entrenamiento
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring="accuracy")
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    metrics = {
        "accuracy": accuracy,
        "f1_score_macro": f1_macro,
        "f1_score_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
    }

    cm = confusion_matrix(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, zero_division=0)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return metrics, cm, report_str, report_dict


def generate_and_save_artifacts(
    cm: np.ndarray,
    labels: List[int],
    report_str: str,
    metrics: Dict[str, float],
    output_dir: Path,
) -> List[str]:
    """Genera artefactos de evaluación (gráficos y reportes) y los guarda en disco.

    Args:
        cm: Matriz de confusión calculada.
        labels: Clases únicas evaluadas.
        report_str: Reporte de clasificación en formato texto.
        metrics: Diccionario de métricas principales.
        output_dir: Directorio donde almacenar los artefactos temporales.

    Returns:
        Lista de rutas absolutas de los archivos generados.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # 1. Matriz de Confusión gráfica
    cm_path = output_dir / "confusion_matrix.png"
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.title("Matriz de Confusión - Clasificación Calidad de Vino", fontsize=14, pad=12)
    plt.xlabel("Clase Predicha", fontsize=12)
    plt.ylabel("Clase Real", fontsize=12)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    generated_files.append(str(cm_path))

    # 2. Reporte de Clasificación en Texto
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== REPORTE DE CLASIFICACIÓN DETALLADO ===\n\n")
        f.write(report_str)
        f.write("\n\n=== MÉTRICAS RESUMEN ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    generated_files.append(str(report_path))

    # 3. Métricas en formato JSON estructurado
    metrics_path = output_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    generated_files.append(str(metrics_path))

    return generated_files


def train_and_track(args: argparse.Namespace) -> str:
    """Función principal de entrenamiento e instrumentación con MLflow.

    Args:
        args: Argumentos parseados de la línea de comandos.

    Returns:
        ID de la corrida de MLflow generada.
    """
    # Configurar el experimento en MLflow
    mlflow.set_experiment(args.experiment_name)

    # 1. Cargar datos
    df = load_data(args.data)

    # 2. Preprocesar datos
    X_train, X_test, y_train, y_test = preprocess_data(
        df=df,
        target_col=args.target,
        test_size=args.test_size,
        random_state=args.seed,
    )

    # 3. Construir pipeline
    pipeline = build_pipeline(
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
        penalty=args.penalty,
        random_state=args.seed,
    )

    # 4. Iniciar corrida de MLflow
    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id

        # Registrar Hiperparámetros con log_param / log_params
        params_to_log = {
            "C": args.C,
            "solver": args.solver,
            "max_iter": args.max_iter,
            "penalty": args.penalty,
            "random_state": args.seed,
            "test_size": args.test_size,
            "scaler": "StandardScaler",
            "model_type": "LogisticRegression",
            "n_samples": len(df),
            "n_features": X_train.shape[1],
        }
        mlflow.log_params(params_to_log)

        # Entrenar pipeline
        pipeline.fit(X_train, y_train)

        # Evaluar modelo
        metrics, cm, report_str, report_dict = evaluate_model(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv_folds=args.cv_folds,
        )

        # Registrar Métricas con log_metric / log_metrics
        mlflow.log_metrics(metrics)

        # Generar y Registrar Artefactos con log_artifact
        artifacts_dir = Path("reports/figures") / run_id
        unique_labels = sorted(list(np.unique(y_test)))
        artifact_files = generate_and_save_artifacts(
            cm=cm,
            labels=unique_labels,
            report_str=report_str,
            metrics=metrics,
            output_dir=artifacts_dir,
        )
        for art_file in artifact_files:
            mlflow.log_artifact(art_file, artifact_path="evaluation_artifacts")

        # Registrar Modelo con mlflow.sklearn.log_model
        input_example = X_train.head(3)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=input_example,
        )

        # Salida informativa en consola
        print("=" * 65)
        print(f" MLFLOW RUN EXITOSA | ID: {run_id}")
        print(f" Experimento: {args.experiment_name}")
        print(f" Corrida: {args.run_name or run_id}")
        print("-" * 65)
        print(" Parámetros:")
        print(f"   --C: {args.C} | --solver: {args.solver} | --max_iter: {args.max_iter} | --penalty: {args.penalty} | --seed: {args.seed}")
        print(" Métricas:")
        print(f"   Accuracy: {metrics['accuracy']:.4f} | F1-Score (Macro): {metrics['f1_score_macro']:.4f} | F1-Score (Weighted): {metrics['f1_score_weighted']:.4f}")
        print(f"   CV-Accuracy: {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
        print(f" Modelo registrado en: runs:/{run_id}/model")
        print("=" * 65)

        return run_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenamiento parametrizado del modelo Wine Quality con MLflow Tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Hiperparámetros requeridos y semilla
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la fuerza de regularización (menor valor = mayor regularización).",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "saga", "liblinear"],
        help="Algoritmo de optimización para LogisticRegression.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        dest="max_iter",
        help="Número máximo de iteraciones permitidas para converger.",
    )
    parser.add_argument(
        "--penalty",
        type=str,
        default="l2",
        choices=["l2", "l1"],
        help="Tipo de regularización (l1 o l2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria (random_state) para asegurar reproducibilidad.",
    )

    # Argumentos de configuración de datos y ejecución
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/winequality-red.csv",
        help="Ruta al archivo CSV del conjunto de datos.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="quality",
        help="Nombre de la variable dependiente / objetivo.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        dest="test_size",
        help="Proporción del conjunto de prueba respecto al total.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        dest="cv_folds",
        help="Número de pliegues para validación cruzada.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="TC5061_Wine_Quality_Experimentation",
        dest="experiment_name",
        help="Nombre descriptivo del experimento en MLflow.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        dest="run_name",
        help="Nombre descriptivo opcional para la corrida.",
    )

    args = parser.parse_args()
    train_and_track(args)


if __name__ == "__main__":
    main()
