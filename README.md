# TC5061 - Experimentación Reutilizable con MLflow (Wine Quality)

**Materia:** TC5061 - Operaciones de Aprendizaje Automático (MLOps)  
**Semana:** 4 - Código de experimentación reutilizable instrumentado con MLFlow  
**Estudiante:** Jesus Ricardo Bahamon Quimbaya  
**Matrícula:** A01840023  
**Profesor:** Dr. Gerardo Reyes  
**Repositorio:** [https://github.com/ricardo-bahamon/TC5061-MLFlow-Experimentation](https://github.com/ricardo-bahamon/TC5061-MLFlow-Experimentation)

---

## 1. Descripción del Proyecto

Este proyecto migra el análisis exploratorio y modelado del conjunto de datos *Wine Quality (Red Wine)* a un flujo de experimentación modular, reproducible y parametrizado por línea de comandos, instrumentado completamente con **MLflow Tracking**.

El script principal (`train.py`) permite que cualquier colaborador o evaluador ejecute el pipeline de entrenamiento modificando hiperparámetros sin tocar el código fuente, garantizando el registro automático de:
1. **Hiperparámetros:** Inverso de regularización (`C`), tipo de penalización (`penalty`), optimizador (`solver`), iteraciones máximas (`max_iter`), y semilla aleatoria (`seed`).
2. **Métricas de Desempeño:** Exactitud global (`accuracy`), macro y weighted $F_1$-score, macro precision, macro recall, y exactitud por validación cruzada estratificada (`cv_accuracy_mean` ± `cv_accuracy_std`).
3. **Artefactos Visuales y de Diagnóstico:** Matriz de confusión en alta resolución (`confusion_matrix.png`), reporte de clasificación por clase (`classification_report.txt`), y resumen de métricas en formato JSON (`metrics_summary.json`).
4. **Modelo Serializado:** Modelo scikit-learn empaquetado bajo el sabor `mlflow.sklearn` con metadata de entorno y firma de entrada/salida.

---

## 2. Estructura del Repositorio (Cookiecutter Data Science)

El proyecto sigue rigurosamente la convención de organización **Cookiecutter Data Science**:

```text
TC5061_MLFlow_Experimentation/
├── Makefile                   # Automatización de tareas (make install, make train, etc.)
├── README.md                  # Documentación integral del repositorio
├── pyproject.toml             # Metadatos del paquete y dependencias modernas
├── setup.py                   # Instalación local del paquete en modo editable
├── requirements.txt           # Dependencias fijadas con versiones exactas
├── .gitignore                 # Reglas de exclusión para Git y MLflow
│
├── data/
│   ├── raw/
│   │   └── winequality-red.csv# Dataset original inmutable de vino tinto (UCI)
│   ├── interim/               # Datos intermedios transformados
│   └── processed/             # Datos listos para entrenamiento
│
├── notebooks/
│   └── Wine_EDA.ipynb         # Notebook conservado exclusivamente para EDA
│
├── reports/
│   └── figures/               # Capturas de pantalla de MLflow y artefactos de corrida
│       ├── runs_table_direct.png
│       ├── mlflow_compare_view.png
│       ├── mlflow_run_detail.png
│       └── mlflow_artifacts_view.png
│
├── src/                       # Código fuente modular reutilizable
│   ├── __init__.py
│   ├── data/
│   │   └── make_dataset.py    # Carga, detección de delimitador y limpieza
│   ├── features/
│   │   └── build_features.py  # División estratificada de variables predictoras
│   └── models/
│       └── train_model.py     # Pipeline con StandardScaler y LogisticRegression
│
├── train.py                   # Script CLI principal de entrenamiento e instrumentación
└── run_experiments.sh         # Script bash que ejecuta la suite de 6 corridas
```

---

## 3. Requisitos e Instalación

### Prerrequisitos
- Python 3.10 o superior (validado en Python 3.13)
- Gestor de paquetes `pip`

### Pasos de instalación
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/ricardo-bahamon/TC5061-MLFlow-Experimentation.git
   cd TC5061-MLFlow-Experimentation
   ```

2. Crear y activar un entorno virtual (recomendado):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # En Linux/macOS
   # .venv\Scripts\activate   # En Windows
   ```

3. Instalar las dependencias fijadas:
   ```bash
   pip install -r requirements.txt
   ```
   *O alternativamente con Make:*
   ```bash
   make install
   ```

---

## 4. Instrucciones de Ejecución

### 4.1. Ejecución con Parámetros por Defecto
El script puede ejecutarse directamente sin argumentos adicionales. Utilizará automáticamente el archivo `data/raw/winequality-red.csv`, el modelo `LogisticRegression` regularizado con $L_2$ ($C=1.0$), el optimizador `lbfgs`, 1000 iteraciones y semilla `42`:

```bash
python train.py
```

### 4.2. Ejecución con Hiperparámetros Personalizados
Puedes sobreescribir cualquier hiperparámetro o configuración desde la terminal. Por ejemplo, para ejecutar la configuración con regularización Lasso ($L_1$) y optimizador `saga`:

```bash
python train.py \
    --experiment-name "TC5061_Wine_Quality_Experimentation" \
    --run-name "run_5_lasso_l1_saga" \
    --C 1.0 \
    --solver saga \
    --penalty l1 \
    --max-iter 1500 \
    --seed 42
```

### 4.3. Ejecutar la Suite Completa de Experimentos
Para reproducir las 6 corridas analizadas en el reporte oficial:

```bash
bash run_experiments.sh
```
*O con Make:*
```bash
make experiments
```

---

## 5. Parámetros de Línea de Comandos (`argparse`)

| Argumento | Tipo | Valor por Defecto | Descripción |
| :--- | :--- | :--- | :--- |
| `--C` | `float` | `1.0` | Inverso de la fuerza de regularización (valores menores indican mayor penalización). |
| `--solver` | `str` | `lbfgs` | Algoritmo de optimización (`lbfgs`, `saga`, `liblinear`). |
| `--penalty` | `str` | `l2` | Norma de regularización (`l2`, `l1`). Nota: `l1` requiere solver `saga` o `liblinear`. |
| `--max-iter` | `int` | `1000` | Número máximo de iteraciones para permitir convergencia de los pesos. |
| `--seed` | `int` | `42` | Semilla del generador pseudoaleatorio (`random_state`) para reproducibilidad estricta. |
| `--data` | `str` | `data/raw/winequality-red.csv` | Ruta al archivo CSV con el conjunto de datos. |
| `--target` | `str` | `quality` | Nombre de la columna objetivo en el dataset. |
| `--test-size` | `float` | `0.2` | Fracción reservada para el conjunto de prueba independiente. |
| `--cv-folds` | `int` | `5` | Pliegues para validación cruzada estratificada sobre el conjunto de entrenamiento. |
| `--experiment-name`| `str` | `TC5061_Wine_Quality_Experimentation` | Nombre del experimento en MLflow donde se agruparán las corridas. |
| `--run-name` | `str` | `None` | Nombre descriptivo opcional para identificar la corrida en la UI. |

---

## 6. Visualización y Comparación en la Interfaz de MLflow

Para lanzar el servidor interactivo de MLflow Tracking:

```bash
mlflow ui --port 5000
```
*O utilizando el target de Make:*
```bash
make mlflow-ui
```

Abre tu navegador en `http://127.0.0.1:5000`:
1. Selecciona el experimento **TC5061_Wine_Quality_Experimentation**.
2. Haz clic en la pestaña **Model training** para ver la tabla comparativa de corridas.
3. Selecciona las casillas de las corridas y pulsa el botón **Compare** para inspeccionar gráficos de coordenadas paralelas, gráficos de dispersión de métricas contra hiperparámetros y diferencias de parámetros lado a lado.
4. Haz clic en cualquier corrida individual para explorar los artefactos registrados (`evaluation_artifacts/confusion_matrix.png`, `classification_report.txt`, `metrics_summary.json`) y el directorio del modelo empaquetado (`model/`).

---

## 7. Resumen de Experimentos Ejecutados

| Corrida | C | Solver | Penalty | Semilla | Exactitud (Test) | $F_1$-Weighted | CV Exactitud |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 1: Baseline** | 1.0 | lbfgs | L2 | 42 | 0.5809 | 0.5541 | 0.5943 ± 0.023 |
| **Run 2: Alta Regularización** | 0.01 | lbfgs | L2 | 42 | 0.5662 | 0.5113 | 0.5658 ± 0.024 |
| **Run 3: Regularización Moderada** | 0.1 | lbfgs | L2 | 42 | 0.5846 | 0.5549 | 0.5842 ± 0.021 |
| **Run 4: Baja Regularización** | 10.0 | lbfgs | L2 | 42 | 0.5809 | 0.5571 | 0.5897 ± 0.027 |
| **Run 5: Lasso L1 SAGA (Mejor)** | 1.0 | saga | L1 | 42 | **0.5919** | **0.5655** | 0.5861 ± 0.024 |
| **Run 6: Réplica Exacta (Seed 42)** | 1.0 | lbfgs | L2 | 42 | 0.5809 | 0.5541 | 0.5943 ± 0.023 |

> **Nota de Reproducibilidad:** Las corridas 1 y 6 presentan resultados idénticos hasta el último decimal, comprobando la estabilidad determinista del pipeline al fijar la semilla aleatoria y estandarizar el preprocesamiento dentro de un Pipeline de scikit-learn.
