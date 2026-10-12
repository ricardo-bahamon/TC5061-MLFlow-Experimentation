#!/bin/bash
set -e

EXPERIMENT="TC5061_Wine_Quality_Experimentation"

echo "=========================================================="
echo " INICIANDO SUITE DE EXPERIMENTACIÓN MLFLOW (TC5061)"
echo " Experimento: $EXPERIMENT"
echo "=========================================================="

echo ">>> Ejecutando Corrida 1: Baseline (C=1.0, lbfgs, penalty=l2, seed=42)..."
python3 train.py \
    --experiment-name "$EXPERIMENT" \
    --run-name "run_1_baseline_C1.0_lbfgs" \
    --C 1.0 \
    --solver lbfgs \
    --penalty l2 \
    --max-iter 1000 \
    --seed 42

echo ">>> Ejecutando Corrida 2: Alta Regularización (C=0.01, lbfgs, penalty=l2, seed=42)..."
python3 train.py \
    --experiment-name "$EXPERIMENT" \
    --run-name "run_2_high_reg_C0.01_lbfgs" \
    --C 0.01 \
    --solver lbfgs \
    --penalty l2 \
    --max-iter 1000 \
    --seed 42

echo ">>> Ejecutando Corrida 3: Regularización Moderada (C=0.1, lbfgs, penalty=l2, seed=42)..."
python3 train.py \
    --experiment-name "$EXPERIMENT" \
    --run-name "run_3_moderate_reg_C0.1_lbfgs" \
    --C 0.1 \
    --solver lbfgs \
    --penalty l2 \
    --max-iter 1000 \
    --seed 42

echo ">>> Ejecutando Corrida 4: Baja Regularización (C=10.0, lbfgs, penalty=l2, seed=42)..."
python3 train.py \
    --experiment-name "$EXPERIMENT" \
    --run-name "run_4_low_reg_C10.0_lbfgs" \
    --C 10.0 \
    --solver lbfgs \
    --penalty l2 \
    --max-iter 1000 \
    --seed 42

echo ">>> Ejecutando Corrida 5: Regularización L1 con Solver SAGA (C=1.0, saga, penalty=l1, seed=42)..."
python3 train.py \
    --experiment-name "$EXPERIMENT" \
    --run-name "run_5_lasso_l1_saga" \
    --C 1.0 \
    --solver saga \
    --penalty l1 \
    --max-iter 1500 \
    --seed 42

echo ">>> Ejecutando Corrida 6: Prueba de Reproducibilidad (Réplica de Corrida 1 con seed=42)..."
python3 train.py \
    --experiment-name "$EXPERIMENT" \
    --run-name "run_6_reproducibility_replica_seed42" \
    --C 1.0 \
    --solver lbfgs \
    --penalty l2 \
    --max-iter 1000 \
    --seed 42

echo "=========================================================="
echo " TODAS LAS CORRIDAS SE COMPLETARON EXITOSAMENTE."
echo "=========================================================="
