#!/bin/bash

# SCRIPT: tuning.sh
# AUTHOR: Samuele Bortolotti <samuele@unitn.it>
# DATE:   2026-01-28
#
# PURPOSE: Hyperparameter tuning 

BASE_OUTPUT_DIR="optuna_runs"
mkdir -p "$BASE_OUTPUT_DIR"

STOP_ON_FAILURE=false 

# MNIST
DATASETS_MNIST=("mnistadd")
DATASETS_MNIST_2=("mnistaddn")
DATASETS_MNIST_FLAGS=("3" "4" "5")
MODELS_MNIST=("lenet")

# CHX
DATASETS_IMG=("chx")
MODELS_IMG=("resnet18 --pretrained")

# RIVAL and CIFAR
DATASETS_CIFAR=("cifar")
MODELS_CIFAR=("resnet18 --pretrained")
DATASETS_RIVAL=("rival")
MODELS_RIVAL=("resnet18")

# CEBAB
DATASETS_CEBAB=("cebab")
MODELS_CEBAB=("bert" "lama")

NESY_VARIANTS=("dpl" "ltn")
CONCEPT_SUPS=("0.0" "1.0")
EPOCHS=20

run_job() {
    local dataset=$1
    local model=$2
    local nesy=$3
    local cs=$4
    local prefix=$5
    local extra_args=$6
    local run_suffix=""

    if [ -n "$prefix" ]; then
        run_suffix="${run_suffix}_${prefix//-/_}"
    fi
    if [ -n "$extra_args" ]; then
        run_suffix="${run_suffix}_${extra_args//-/_}"
    fi

    run_suffix=$(echo "$run_suffix" | sed 's/__*/_/g')
    RUN_NAME="${dataset}_${model}_${nesy}_cs${cs//./}${run_suffix}"

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"
    mkdir -p "$OUTPUT_DIR"

    echo "Starting run: $RUN_NAME"

    CMD=(
        python -m conformal CONF --dry-run optuna
        --optuna-path "$OUTPUT_DIR"
        --concept-sup "$cs"
        --epochs "$EPOCHS"
        "$dataset"
    )

    if [ -n "$prefix" ]; then CMD+=("$prefix"); fi
    if [ -n "$extra_args" ]; then CMD+=("$extra_args"); fi

    CMD+=("$model" "$nesy")

    echo "Executing command:"
    echo "${CMD[*]}"

    "${CMD[@]}" > "${OUTPUT_DIR}/log.txt" 2>&1

    local STATUS=$?
    if [ -ne 0 ]; then
        echo "Run FAILED: $RUN_NAME (see ${OUTPUT_DIR}/log.txt)"
        if [ "$STOP_ON_FAILURE" = true ]; then
            echo "Stopping script due to failure."
            exit $STATUS
        fi
    else
        echo "Run finished successfully: $RUN_NAME"
        echo "DONE" > "${OUTPUT_DIR}/finished.txt"
    fi
}

echo "[INFO] Starting jobs..."
echo "[INFO] Starting MNIST..."

# MNIST
for dataset in "${DATASETS_MNIST[@]}"; do
    for model in "${MODELS_MNIST[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                run_job "$dataset" "$model" "$nesy" "$cs" "" ""
            done
        done
    done
done

echo "[INFO] MNIST done!"
echo "[INFO] Starting MNIST-ADDN..."

# MNIST-ADDN
for dataset in "${DATASETS_MNIST_2[@]}"; do
    for model in "${MODELS_MNIST[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                for nd in "${DATASETS_MNIST_FLAGS[@]}"; do
                    run_job "$dataset" "$model" "$nesy" "$cs" "--n-digits" "$nd"
                done
            done
        done
    done
done

echo "[INFO] MNIST-ADDN done!"
echo "[INFO] Starting CHX..."

# CHX
for dataset in "${DATASETS_IMG[@]}"; do
    for model in "${MODELS_IMG[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                run_job "$dataset" "$model" "$nesy" "$cs" "--chx-multi-class" ""
            done
        done
    done
done

echo "[INFO] CHX done!"
echo "[INFO] Starting CIFAR..."

# CIFAR
for dataset in "${DATASETS_CIFAR[@]}"; do
    for model in "${MODELS_CIFAR[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                run_job "$dataset" "$model" "$nesy" "$cs" "" ""
            done
        done
    done
done

echo "[INFO] CIFAR done!"
echo "[INFO] Starting RIVAL..."

# RIVAL
for dataset in "${DATASETS_RIVAL[@]}"; do
    for model in "${MODELS_RIVAL[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                run_job "$dataset" "$model" "$nesy" "$cs" "" ""
            done
        done
    done
done

echo "[INFO] RIVAL done!"
echo "[INFO] Starting CEBAB..."

# CEBAB
for dataset in "${DATASETS_CEBAB[@]}"; do
    for model in "${MODELS_CEBAB[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                run_job "$dataset" "$model" "$nesy" "$cs" "" ""
            done
        done
    done
done

echo "[INFO] CEBAB done!"
echo "[INFO] Starting DERMA..."

# DERMA
for model in "${MODELS_IMG[@]}"; do
    for nesy in "${NESY_VARIANTS[@]}"; do
        for cs in "${CONCEPT_SUPS[@]}"; do
            run_job "derma" "$model" "$nesy" "$cs" "" ""
        done
    done
done

echo "[INFO] DERMA done!"
echo "All runs completed."