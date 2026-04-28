#!/bin/bash

# SCRIPT: check.sh
# AUTHOR: Samuele Bortolotti <samuele@unitn.it>
# DATE:   2026-02-01
#
# PURPOSE: Check no crashes happen (2 epochs)

EPOCHS=2
NESY_VARIANTS=("dpl" "ltn")
CONCEPT_SUPS=("0.0" "1.0")

# MNIST
DATASETS_MNIST=("mnistadd" "mnisthalf" "mnistsump" "mnistaddn --n-digits 4")
MODELS_MNIST=("lenet")

# CHX / DERMA
DATASETS_IMG=("chx --chx-multi-class" "derma")
MODELS_IMG=("resnet18 --pretrained")
# CIFAR
DATASETS_CIFAR=("cifar")
MODELS_CIFAR=("resnet18 --pretrained")
# RIVAL
DATASETS_RIVAL=("rival")
MODELS_RIVAL=("resnet18")
# CEBAB
DATASETS_CEBAB=("cebab")
MODELS_CEBAB=("bert" "lama")
# BOIA
DATASETS_BOIA=("boia")
MODELS_BOIA=("linear")


# --- 2. EXECUTION FUNCTION ---
run_job() {
    local dataset=$1
    local model=$2
    local nesy=$3
    local cs=$4

    CMD=(
        python -m conformal CONF --dry-run train
        --concept-sup "$cs"
        --epochs "$EPOCHS"
    )

    CMD+=($dataset $model "$nesy")

    echo "Executing command:"
    echo "${CMD[*]}"

    "${CMD[@]}"

    local STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "[ERROR] Command FAILED: ${CMD[*]}"
    else
        echo "[SUCCESS] Run finished successfully: ${CMD[*]}"
        echo "---------------------------------------------------"
    fi
}


echo "[INFO] Starting jobs..."

# MNIST
echo "[INFO] Starting MNIST..."
for dataset in "${DATASETS_MNIST[@]}"; do
    for model in "${MODELS_MNIST[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                run_job "$dataset" "$model" "$nesy" "$cs"
            done
        done
    done
done
echo "[INFO] MNIST done!"

# CHX / DERMA
echo "[INFO] Starting CHX / DERMA..."
for dataset in "${DATASETS_IMG[@]}"; do
    for model in "${MODELS_IMG[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                run_job "$dataset" "$model" "$nesy" "$cs"
            done
        done
    done
done
echo "[INFO] CHX / DERMA done!"

# CIFAR
echo "[INFO] Starting CIFAR..."
for dataset in "${DATASETS_CIFAR[@]}"; do
    for model in "${MODELS_CIFAR[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                run_job "$dataset" "$model" "$nesy" "$cs"
            done
        done
    done
done
echo "[INFO] CIFAR done!"

# RIVAL
echo "[INFO] Starting RIVAL..."
for dataset in "${DATASETS_RIVAL[@]}"; do
    for model in "${MODELS_RIVAL[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                run_job "$dataset" "$model" "$nesy" "$cs"
            done
        done
    done
done
echo "[INFO] RIVAL done!"

# CEBAB
echo "[INFO] Starting CEBAB..."
for dataset in "${DATASETS_CEBAB[@]}"; do
    for model in "${MODELS_CEBAB[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                run_job "$dataset" "$model" "$nesy" "$cs"
            done
        done
    done
done
echo "[INFO] CEBAB done!"

# BOIA
echo "[INFO] Starting BOIA..."
for dataset in "${DATASETS_BOIA[@]}"; do
    for model in "${MODELS_BOIA[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                run_job "$dataset" "$model" "$nesy" "$cs"
            done
        done
    done
done
echo "[INFO] BOIA done!"

echo "[INFO] All runs completed."