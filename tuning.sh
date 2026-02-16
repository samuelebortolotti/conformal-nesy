#!/bin/bash

# SCRIPT: tuning.sh
# AUTHOR: Samuele Bortolotti <samuele@unitn.it>
# DATE:   2026-01-28
#
# PURPOSE: Hyperparameter tuning 

BASE_OUTPUT_DIR="optuna_runs"
mkdir -p "$BASE_OUTPUT_DIR"

STOP_ON_FAILURE=true 

DATASETS_MNIST=("mnistadd") # "mnistadd" 
DATASETS_MNIST_2=("mnistaddn")
DATASETS_MNIST_FLAGS=("--n-digits 3" "--n-digits 4" "--n-digits 5")
MODELS_MNIST=("lenet")
DATASETS_IMG=("chx") #  "derma"
MODELS_IMG=("resnet18 --pretrained")
DATASETS_BOIA=("boia")
MODELS_BOIA=("linear")
NESY_VARIANTS=("dpl" "ltn") # "dsl" "linpred")
CONCEPT_SUPS=("0.0" "1.0")
EPOCHS=20

run_job() {
    local dataset=$1
    local model=$2
    local nesy=$3
    local cs=$4
    local extra_args=$5

    RUN_NAME="${dataset}_${model}_${nesy}_cs${cs//./}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"
    mkdir -p "$OUTPUT_DIR"

    echo "Starting run: $RUN_NAME"

    CMD="python -m conformal \
        CONF --dry-run optuna \
        --optuna-path $OUTPUT_DIR \
        --concept-sup $cs \
        --epochs $EPOCHS \
        $dataset $extra_args \
        $model $nesy \
        > \"${OUTPUT_DIR}/log.txt\" 2>&1"

    echo "Executing command:"
    echo "$CMD"

    eval $CMD

    # check on the status
    local STATUS=$?
    if [ $STATUS -ne 0 ]; then
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
echo "[INFO} Starting MNIST-N..."

# MNIST
for dataset in "${DATASETS_MNIST_2[@]}"; do
    for model in "${MODELS_MNIST[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                for nd in "${DATASETS_MNIST_FLAGS[@]}"; do
                    run_job "$dataset" "$model" "$nesy" "$cs" "$nd"
                done
            done
        done
    done
done

echo "[INFO] MNIST-ADDN done!"
echo "[INFO} Starting CHX..."

# CHX / DERMA
for dataset in "${DATASETS_IMG[@]}"; do
    for model in "${MODELS_IMG[@]}"; do
        for nesy in "${NESY_VARIANTS[@]}"; do
            for cs in "${CONCEPT_SUPS[@]}"; do
                run_job "$dataset" "$model" "$nesy" "$cs" "--chx-multi-class"
            done
        done
    done
done

echo "[INFO] CHX done!"

# echo "[INFO} Starting BOIA..."

# # BOIA
# for dataset in "${DATASETS_BOIA[@]}"; do
#     for model in "${MODELS_BOIA[@]}"; do
#         for nesy in "${NESY_VARIANTS[@]}"; do
#             for cs in "${CONCEPT_SUPS[@]}"; do
#                 run_job "$dataset" "$model" "$nesy" "$cs" ""
#             done
#         done
#     done
# done

# echo "[INFO] BOIA done!"
# echo "All runs completed."