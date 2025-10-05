#!/usr/bin/env bash
###############################################################################
# vllm_bench.sh
# vLLM benchmark orchestrator:
#  - Global + per-model env vars
#  - Global + per-model configs (parallel/scheduler/eplb)
#  - Model-signature hashing → unique per-config subdirs
#  - Nsight Systems: nsys launch (server) + nsys start/stop (per concurrency)
#  - Study dir with manifest + global summary + run log
#  - Per-model subdirs with logs/results/profiles + per-model summary
###############################################################################
set -euo pipefail

CONFIG_FILE=${1:? "Usage: vllm_bench.sh <config.yaml>"}

###############################################################################
# 1) Dependencies
###############################################################################
if ! command -v jq &>/dev/null; then
  if command -v apt-get &>/dev/null; then sudo apt-get update && sudo apt-get install -y jq
  elif command -v dnf &>/dev/null; then sudo dnf install -y jq
  else echo "❌ jq not found. Please install jq."; exit 1; fi
fi
if ! command -v yq &>/dev/null; then
  # Python wrapper yq
  pip install --quiet yq || { echo "❌ Failed to install yq"; exit 1; }
fi
###############################################################################
# 2) Generate vllm congig yamls 
###############################################################################
# In vllm_bench.sh (setup section)
if [[ "${GENERATE_CONFIG_TEMPLATES:-false}" == "true" ]]; then
  ./generate_vllm_config_templates.sh
fi


###############################################################################
# 2) Parse global bench config
###############################################################################
INPUT_LEN=$(python3 -m yq -r '.bench.input_len' < "$CONFIG_FILE")
OUTPUT_LEN=$(python3 -m yq -r '.bench.output_len' < "$CONFIG_FILE")
CC_MULT=$(python3 -m yq -r '.bench.cc_mult' < "$CONFIG_FILE")
CONCURRENCIES=($(python3 -m yq -r '.bench.concurrencies[]' < "$CONFIG_FILE"))
RESULT_PREFIX=$(python3 -m yq -r '.bench.result_prefix' < "$CONFIG_FILE")
COLLECT_NSYS=$(python3 -m yq -r '.bench.collect_nsys // "false"' < "$CONFIG_FILE")
GLOBAL_VLLM_BENCH_OPTS=$(python3 -m yq -r '.bench.vllm_bench_opts // ""' < "$CONFIG_FILE")
GLOBAL_PROFILE=$(python3 -m yq -r '.bench.profile // "false"' < "$CONFIG_FILE")

# Global configs (optional; per-model may override)
GLOBAL_PARALLEL=$(python3 -m yq -r '.bench.configs.parallel // ""' < "$CONFIG_FILE")
GLOBAL_SCHEDULER=$(python3 -m yq -r '.bench.configs.scheduler // ""' < "$CONFIG_FILE")
GLOBAL_EPLB=$(python3 -m yq -r '.bench.configs.eplb // ""' < "$CONFIG_FILE")

###############################################################################
# 3) Study directory + run log + manifest
###############################################################################
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
USER_SPECIFIED_DIR=$(python3 -m yq -r '.bench.study_dir // ""' < "$CONFIG_FILE")
STUDY_DIR=${USER_SPECIFIED_DIR:-"Study_${TIMESTAMP}"}
mkdir -p "$STUDY_DIR"
STUDY_LOG="${STUDY_DIR}/run_output.log"
SUMMARY_CSV="${STUDY_DIR}/summary.csv"

# Redirect all stdout/stderr to both console and study log
exec > >(tee -a "$STUDY_LOG") 2>&1

echo "📁 Study directory: $STUDY_DIR"
mkdir -p "$STUDY_DIR"/{profiles,results,logs} # top-level (kept for convenience)

# Apply GLOBAL environment variables (bench.env)
NUM_GLOBAL_ENV=$(python3 -m yq -r '.bench.env | length' < "$CONFIG_FILE" 2>/dev/null || echo 0)
if [[ "$NUM_GLOBAL_ENV" -gt 0 ]]; then
  echo "🌍 Applying global environment variables"
  while IFS= read -r key; do
    val=$(python3 -m yq -r ".bench.env[\"$key\"]" < "$CONFIG_FILE")
    echo "  export $key=$val"; export "$key=$val"
  done < <(python3 -m yq -r ".bench.env | keys_unsorted[]" < "$CONFIG_FILE")
fi

# Manifest (lightweight; enough for provenance)
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo "N/A")
NUM_MODELS=$(python3 -m yq -r '.models | length' < "$CONFIG_FILE")
cat > "${STUDY_DIR}/study_manifest.json" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "study_dir": "${STUDY_DIR}",
  "gpu": "${GPU_INFO}",
  "num_models": ${NUM_MODELS},
  "config_snapshot": $(jq -Rs . < "$CONFIG_FILE")
}
EOF

# Global summary header
echo "model,model_sig,concurrency,num_prompts,status,runtime_sec,result_file,log_file,nsys_file" > "$SUMMARY_CSV"

###############################################################################
# 4) Model execution loop (per-model subdirs + env + profiling + summaries)
###############################################################################
for ((i=0; i<NUM_MODELS; i++)); do
  MODEL=$(python3 -m yq -r ".models[$i].name" < "$CONFIG_FILE")
  PORT=$(python3 -m yq -r ".models[$i].port" < "$CONFIG_FILE")
  PARAMS=$(python3 -m yq -r ".models[$i].params // \"\"" < "$CONFIG_FILE")

  # Merge compilation_config if provided (quote JSON to keep it single arg)
  if [[ "$(python3 -m yq -r ".models[$i] | has(\"compilation_config\")" < "$CONFIG_FILE")" == "true" ]]; then
    COMPILATION=$(python3 -m yq -r ".models[$i].compilation_config" < "$CONFIG_FILE" | jq -c .)
    PARAMS="$PARAMS --compilation-config '$COMPILATION'"
  fi

  # Merge per-model configs (override global). Quote to preserve as single token.
  MODEL_PARALLEL=$(python3 -m yq -r ".models[$i].configs.parallel // \"$GLOBAL_PARALLEL\"" < "$CONFIG_FILE")
  MODEL_SCHEDULER=$(python3 -m yq -r ".models[$i].configs.scheduler // \"$GLOBAL_SCHEDULER\"" < "$CONFIG_FILE")
  MODEL_EPLB=$(python3 -m yq -r ".models[$i].configs.eplb // \"$GLOBAL_EPLB\"" < "$CONFIG_FILE")
  [[ -n "$MODEL_PARALLEL" && "$MODEL_PARALLEL" != "null" ]] && PARAMS="$PARAMS --parallel-config '$MODEL_PARALLEL'"
  [[ -n "$MODEL_SCHEDULER" && "$MODEL_SCHEDULER" != "null" ]] && PARAMS="$PARAMS --scheduler-config '$MODEL_SCHEDULER'"
  [[ -n "$MODEL_EPLB" && "$MODEL_EPLB" != "null" ]] && PARAMS="$PARAMS --eplb-config '$MODEL_EPLB'"

  # Per-model env vars (override/extend global)
  NUM_ENV=$(python3 -m yq -r ".models[$i].env | length" < "$CONFIG_FILE" 2>/dev/null || echo 0)
  if [[ "$NUM_ENV" -gt 0 ]]; then
    echo "🌍 Applying env vars for $MODEL"
    while IFS= read -r key; do
      val=$(python3 -m yq -r ".models[$i].env[\"$key\"]" < "$CONFIG_FILE")
      echo "  export $key=$val"; export "$key=$val"
    done < <(python3 -m yq -r ".models[$i].env | keys_unsorted[]" < "$CONFIG_FILE")
  fi

  # Per-model overrides and profiling args
  MODEL_VLLM_BENCH_OPTS=$(python3 -m yq -r ".models[$i].vllm_bench_opts // \"$GLOBAL_VLLM_BENCH_OPTS\"" < "$CONFIG_FILE")
  MODEL_PROFILE=$(python3 -m yq -r ".models[$i].profile // \"$GLOBAL_PROFILE\"" < "$CONFIG_FILE")
  MODEL_RESULT_FILE=$(python3 -m yq -r ".models[$i].result_file // \"${RESULT_PREFIX}_${MODEL//\//_}.json\"" < "$CONFIG_FILE")
  NSYS_LAUNCH_ARGS=$(python3 -m yq -r ".models[$i].profiling.nsys_launch_args // \"\"" < "$CONFIG_FILE")
  NSYS_START_ARGS=$(python3 -m yq -r ".models[$i].profiling.nsys_start_args // \"\"" < "$CONFIG_FILE")

  # Build model signature (unique per config) and subdirectories
  SIG_SRC="${MODEL}|${PORT}|${PARAMS}|${MODEL_PARALLEL}|${MODEL_SCHEDULER}|${MODEL_EPLB}|${NSYS_LAUNCH_ARGS}|${NSYS_START_ARGS}"
  MODEL_SIG=$(echo -n "$SIG_SRC" | md5sum | cut -c1-8)
  MODEL_DIR="${STUDY_DIR}/model_${MODEL//\//_}_${MODEL_SIG}"
  mkdir -p "${MODEL_DIR}"/{logs,results,profiles}

  # File paths (per-model)
  LOGFILE="${MODEL_DIR}/logs/vllm_server.log"
  RESULT_PATH="${MODEL_DIR}/results/${MODEL_RESULT_FILE}"
  PER_MODEL_SUMMARY="${MODEL_DIR}/summary_model.csv"
  # Per-model summary header
  echo "model,model_sig,concurrency,num_prompts,status,runtime_sec,result_file,log_file,nsys_file" > "$PER_MODEL_SUMMARY"

  echo "=============================================="
  echo "🚀 Model: $MODEL"
  echo "🔑 Signature: $MODEL_SIG"
  echo "🌐 Port: $PORT"
  echo "⚙️  Params: $PARAMS"
  echo "📂 Model dir: $MODEL_DIR"
  echo "=============================================="

  # Per-model manifest
  cat > "${MODEL_DIR}/manifest_model.json" <<EOF
{
  "model": "${MODEL}",
  "model_signature": "${MODEL_SIG}",
  "port": "${PORT}",
  "params": ${PARAMS:+$(jq -Rs . <<< "$PARAMS")},
  "configs": {
    "parallel": ${MODEL_PARALLEL:+$(jq -Rs . <<< "$MODEL_PARALLEL")},
    "scheduler": ${MODEL_SCHEDULER:+$(jq -Rs . <<< "$MODEL_SCHEDULER")},
    "eplb": ${MODEL_EPLB:+$(jq -Rs . <<< "$MODEL_EPLB")}
  },
  "profiling": {
    "launch_args": ${NSYS_LAUNCH_ARGS:+$(jq -Rs . <<< "$NSYS_LAUNCH_ARGS")},
    "start_args": ${NSYS_START_ARGS:+$(jq -Rs . <<< "$NSYS_START_ARGS")}
  },
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

  # Launch vLLM (optionally under Nsight Systems "launch" mode)
  if [[ "$MODEL_PROFILE" == "true" || "$COLLECT_NSYS" == "true" ]]; then
    NSYS_LAUNCH_FILE="${MODEL_DIR}/profiles/nsys_vllm_server"
    echo "▶️ nsys launch → ${NSYS_LAUNCH_FILE}.qdrep (attached; not recording yet)"
    nsys launch ${NSYS_LAUNCH_ARGS} \
      --output "${NSYS_LAUNCH_FILE}" \
      vllm serve "$MODEL" --port "$PORT" $PARAMS >"$LOGFILE" 2>&1 &
  else
    setsid vllm serve "$MODEL" --port "$PORT" $PARAMS >"$LOGFILE" 2>&1 &
  fi

  VLLM_PID=$!
  echo "PID=$VLLM_PID"

  # Wait for server readiness
  echo "⏳ Waiting for vLLM on port $PORT..."
  for j in {1..10}; do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null; then
      echo "✅ vLLM is ready."; break
    fi
    sleep 5
  done

  # Bench loop (per concurrency) with optional per-iteration Nsight capture
  for CONCURRENCY in "${CONCURRENCIES[@]}"; do
    NUM_PROMPTS=$((CONCURRENCY * CC_MULT))
    STATUS="success"
    START_TS=$(date +%s)
    NSYS_FILE=""

    echo ""
    echo "===== $MODEL (sig $MODEL_SIG): $NUM_PROMPTS prompts @ conc $CONCURRENCY ====="

    if [[ "$MODEL_PROFILE" == "true" || "$COLLECT_NSYS" == "true" ]]; then
      NSYS_FILE="${MODEL_DIR}/profiles/nsys_conc${CONCURRENCY}"
      echo "🎥 nsys start → ${NSYS_FILE}.qdrep"
      nsys start ${NSYS_START_ARGS} --output "$NSYS_FILE"
    fi

    if ! vllm bench serve \
      --base-url "http://localhost:$PORT" \
      --model "$MODEL" \
      --dataset-name random \
      --random-input-len "$INPUT_LEN" \
      --random-output-len "$OUTPUT_LEN" \
      --max-concurrency "$CONCURRENCY" \
      --num-prompts "$NUM_PROMPTS" \
      --seed "$(date +%s)" \
      --save-result \
      --result-filename "$RESULT_PATH" \
      --append-result \
      $MODEL_VLLM_BENCH_OPTS; then
        STATUS="failed"
    fi

    if [[ "$MODEL_PROFILE" == "true" || "$COLLECT_NSYS" == "true" ]]; then
      echo "🛑 nsys stop"
      nsys stop
      echo "✅ Nsight report saved: ${NSYS_FILE}.qdrep"
    fi

    END_TS=$(date +%s)
    RUNTIME=$((END_TS - START_TS))

    # Append per-model and global summaries
    echo "$MODEL,$MODEL_SIG,$CONCURRENCY,$NUM_PROMPTS,$STATUS,$RUNTIME,$RESULT_PATH,$LOGFILE,$NSYS_FILE" >> "$PER_MODEL_SUMMARY"
    echo "$MODEL,$MODEL_SIG,$CONCURRENCY,$NUM_PROMPTS,$STATUS,$RUNTIME,$RESULT_PATH,$LOGFILE,$NSYS_FILE" >> "$SUMMARY_CSV"
  done

  echo "🛑 Stopping vLLM server for $MODEL (sig $MODEL_SIG)"
  pkill -TERM -P ${VLLM_PID} 2>/dev/null || true
  kill -TERM ${VLLM_PID} 2>/dev/null || true
  sleep 3
done

###############################################################################
# 5) Final summary
###############################################################################
echo ""
echo "🎉 All benchmarking completed."
echo "📊 Global summary:  $SUMMARY_CSV"
echo "🧾 Study manifest:  ${STUDY_DIR}/study_manifest.json"
echo "📜 Run log:         ${STUDY_DIR}/run_output.log"
echo "📂 Per-model dirs:  ${STUDY_DIR}/model_*"


