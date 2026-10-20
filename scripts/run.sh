set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-iemocap}"
FEATURE_ROOT="${FEATURE_ROOT:-${PROJECT_DIR}/data}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local Qwen2.5-7B-Instruct directory}"
DEVICE="${DEVICE:-cuda:0}"
WORK="${WORK:-${PROJECT_DIR}/outputs/${DATASET}}"
if [[ "${DATASET}" != "iemocap" && "${DATASET}" != "meld" ]]; then
    echo 'DATASET must be iemocap or meld' >&2
    exit 2
fi
cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
PARTITION_ARGS=()
if [[ "${DATASET}" == "meld" ]]; then
    PARTITION_ARGS=(--partitions "${PARTITIONS:-${FEATURE_ROOT}/meld_features_roberta.pkl}")
fi
"${PYTHON_BIN}" -m ecra.prepare splits --config "configs/${DATASET}.json" --features "${FEATURE_ROOT}/${DATASET}_multimodal_features.pkl" "${PARTITION_ARGS[@]}" --output "${WORK}"
"${PYTHON_BIN}" -m ecra.train pairs --work "${WORK}" --device "${DEVICE}"
"${PYTHON_BIN}" -m ecra.prepare prompts --config "configs/${DATASET}.json" --features "${FEATURE_ROOT}/${DATASET}_multimodal_features.pkl" "${PARTITION_ARGS[@]}" --pairs "${WORK}/pairs.json" --output "${WORK}"
"${PYTHON_BIN}" -m ecra.prepare generate --work "${WORK}" --model-path "${MODEL_PATH}" --device "${DEVICE}" --resume
"${PYTHON_BIN}" -m ecra.prepare calibrate --work "${WORK}"
"${PYTHON_BIN}" -m ecra.train fit --work "${WORK}" --device "${DEVICE}"
