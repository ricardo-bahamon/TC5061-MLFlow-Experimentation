"""Módulo de ingestión y carga de datos."""
from pathlib import Path
import pandas as pd


def load_raw_wine_data(filepath: str = "data/raw/winequality-red.csv") -> pd.DataFrame:
    """Carga y limpia el dataset crudo de calidad de vino tinto.

    Detecta el delimitador (coma o punto y coma), normaliza nombres de columnas
    a snake_case y elimina duplicados o nulos.
    """
    path = Path(filepath)
    if not path.exists():
        candidate_paths = [
            path,
            Path("data/raw/winequality-red.csv"),
            Path("../data/raw/winequality-red.csv"),
        ]
        for c in candidate_paths:
            if c.exists():
                path = c
                break

    if not path.exists():
        raise FileNotFoundError(f"No se localizó el archivo de datos en: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        delimiter = ";" if ";" in first_line else ","

    df = pd.read_csv(path, sep=delimiter)
    df.columns = [c.strip().lower().replace(" ", "_").replace('"', "") for c in df.columns]
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    return df
