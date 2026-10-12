.PHONY: help install train experiments mlflow-ui clean

PYTHON = python3
PIP = pip3

help:
	@echo "========================================================================"
	@echo " TC5061 - Experimentación con MLflow (Wine Quality)"
	@echo "========================================================================"
	@echo " Comandos disponibles:"
	@echo "   make install       - Instala las dependencias del proyecto"
	@echo "   make train         - Ejecuta el script de entrenamiento con valores por defecto"
	@echo "   make experiments   - Ejecuta la suite de 6 experimentos parametrizados"
	@echo "   make mlflow-ui     - Inicia el servidor de MLflow UI en http://127.0.0.1:5000"
	@echo "   make clean         - Limpia archivos temporales y bytecode de Python"
	@echo "========================================================================"

install:
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) train.py

experiments:
	bash run_experiments.sh

mlflow-ui:
	mlflow ui --port 5000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
