#!/bin/bash

# SCRIPT: check.sh
# AUTHOR: Samuele Bortolotti <samuele@unitn.it>
# DATE:   2026-02-01
#
# PURPOSE: Check no crashes happens

DATASETS_MNIST=("mnistadd" "mnisthalf" "mnistsump", "mnistaddn --n-digits 4")
MODELS_MNIST=("lenet")
DATASETS_IMG=("chx" "derma")
MODELS_IMG=("resnet18 --pretrained")
DATASETS_BOIA=("boia")
MODELS_BOIA=("linear")
NESY_VARIANTS=("dpl" "ltn" "dsl" "linpred")
CONCEPT_SUPS=("0.0" "1.0")
EPOCHS=2

run_job() {
    local dataset=$1
    local model=$2
    local nesy=$3
    local cs=$4
    local extra_args=$5

    CMD="python -m conformal \
        CONF --dry-run train \
        --concept-sup $cs \
        --epochs $EPOCHS \
        $dataset $extra_args \
        $model $nesy"

    echo "Executing command:"
    echo "$CMD"

    eval $CMD

    # check on the status
    local STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "Command FAILED: $CMD"
    else
        echo "Run finished successfully: $CMD"
    fi
}

echo "[INFO} Starting jobs..."
echo "[INFO} Starting MNIST..."

# MNIST
for dataset in "${DATASETS_MNIST[@]}"; do
    for model in "${MODELS_MNIST[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                run_job "$dataset" "$model" "$nesy" "$cs" ""
            done
        done
    done
done

echo "[INFO] MNIST done!"
echo "[INFO} Starting CHX / DERMA..."

# CHX / DERMA
for dataset in "${DATASETS_IMG[@]}"; do
    for model in "${MODELS_IMG[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                run_job "$dataset" "$model" "$nesy" "$cs" ""
            done
        done
    done
done

echo "[INFO] CHX / DERMA done!"
echo "[INFO} Starting BOIA..."

# BOIA
for dataset in "${DATASETS_BOIA[@]}"; do
    for model in "${MODELS_BOIA[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                run_job "$dataset" "$model" "$nesy" "$cs" ""
            done
        done
    done
done

echo "[INFO] BOIA done!"
echo "All runs completed."