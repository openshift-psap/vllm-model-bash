#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=$1
SUMMARY_CSV="summary.csv"

# --- Ensure HF_HOME is set ---
if [[ -z "${HF_HOME:-}" ]]; then
  echo "❌ HF_HOME is not set!"
  echo "Please set it before running this script, e.g.:"
  echo "  export HF_HOME=/mnt/workspace"
  exit 1
else
  echo "✅ HF_HOME is set to: $HF_HOME"
fi

# --- Ensure jq is installed ---
if ! command -v jq &>/dev/null; then
  echo "⚠️ jq not found, trying to install..."
  if command -v apt-get &>/dev/null; then
    apt-get update && apt-get install -y jq
  elif command -v dnf &>/dev/null; then
    dnf install -y jq
  else
    echo "❌ jq is not installed and cannot be auto-installed. Please install manually."
    exit 1
  fi
fi

# --- Ensure yq (Python wrapper) is installed ---
if ! command -v yq &>/dev/null; then
  echo "⚠️ yq not found, installing via pip..."
  pip install --quiet yq
fi

echo "✅ Using yq: $(yq --version)"
echo "✅ Using jq: $(jq --version)"

# --- Extract bench parameters ---
INPUT_LEN=$(python3 -m yq -r '.bench.input_len' < "$CONFIG_FILE")
OUTPUT_LEN=$(python3 -m yq -r '.bench.output_len' < "$CONFIG_FILE")
CC_MULT=$(python3 -m yq -r '.bench.cc_mult' < "$CONFIG_FILE")
CONCURRENCIES=($(python3 -m yq -r '.bench.concurrencies[]' < "$CONFIG_FILE"))
RESULT_PREFIX=$(python3 -m yq -r '.bench.result_prefix' < "$CONFIG_FILE")
COLLECT_NSYS=$(python3 -m yq -r '.bench.collect_nsys' < "$CONFIG_FILE")

# --- Prepare summary CSV ---
echo "model,concurrency,num_prompts,status,runtime_sec,result_file,log_file,nsys_file" > "$SUMMARY_CSV"

# --- Loop over models ---
NUM_MODELS=$(python3 -m yq -r '.models | length' < "$CONFIG_FILE")

for ((i=0; i<NUM_MODELS; i++)); do
  MODEL=$(python3 -m yq -r ".models[$i].name" < "$CONFIG_FILE")
  PORT=$(python3 -m yq -r ".models[$i].port" < "$CONFIG_FILE")
  PARAMS=$(python3 -m yq -r ".models[$i].params" < "$CONFIG_FILE")


  # Check if compilation_config exists
if python3 -m yq -r ".models[$i] | has(\"compilation_config\")" < "$CONFIG_FILE" | grep -q true; then
  COMPILATION=$(python3 -m yq -r ".models[$i].compilation_config" < "$CONFIG_FILE" | jq -c .)
  PARAMS="$PARAMS --compilation-config $COMPILATION"
fi
  #COMPILATION=$(python3 -m yq -r ".models[$i].compilation_config" < "$CONFIG_FILE")
  #PARAMS="$PARAMS --compilation-config $COMPILATION"


  # --- Export model-specific env vars ---
  NUM_ENV=$(python3 -m yq -r ".models[$i].env | length" < "$CONFIG_FILE" 2>/dev/null || echo 0)
  if [[ "$NUM_ENV" -gt 0 ]]; then
  	echo "🌍 Setting env vars for $MODEL"
  	for key in $(python3 -m yq -r ".models[$i].env | keys_unsorted[]" < "$CONFIG_FILE"); do
    		val=$(python3 -m yq -r ".models[$i].env[\"$key\"]" < "$CONFIG_FILE")
    		echo "  export $key=$val"
    		export $key="$val"
  	done
  fi


  echo "=============================================="
  echo "🚀 Launching vLLM for model $MODEL on port $PORT"
  echo "Params: $PARAMS"
  echo "=============================================="

  LOGFILE="vllm_${MODEL//\//_}.log"

  if [[ "$COLLECT_NSYS" == "true" ]]; then
  	NSYS_LAUNCH_FILE="nsys_vllm_${MODEL//\//_}_server"
	NSYS_LAUNCH_ARGS=$(python3 -m yq -r ".models[$i].profiling.nsys_launch_args // \"\"" < "$CONFIG_FILE")
  	echo "▶️ Launching vLLM under Nsight Systems: $NSYS_LAUNCH_FILE"
  	nsys launch ${NSYS_LAUNCH_ARGS} \
    	vllm serve "$MODEL" --port "$PORT" $PARAMS >"$LOGFILE" 2>&1 &
  else
  	setsid vllm serve "$MODEL" --port "$PORT" $PARAMS >"$LOGFILE" 2>&1 &
  fi

  VLLM_PID=$!
  PGID=$(ps -o pgid= $VLLM_PID | tr -d ' ')
  echo "PID=$VLLM_PID, PGID=$PGID"

  # Wait for server ready
  echo "⏳ Waiting for vLLM on port $PORT..."
  for j in {1..4}; do
    if curl -s "http://localhost:$PORT/v1/models" >/dev/null; then
      echo "✅ vLLM is ready."
      break
    fi
    sleep 5
  done

  # Benchmark loop
  OUTFILE="${RESULT_PREFIX}_${MODEL//\//_}.json"
  for CONCURRENCY in "${CONCURRENCIES[@]}"; do
    NUM_PROMPTS=$(($CONCURRENCY * $CC_MULT))
    echo ""
    echo "===== $MODEL: $NUM_PROMPTS prompts @ concurrency $CONCURRENCY ====="
    echo ""

    STATUS="success"
    START_TS=$(date +%s)
    NSYS_FILE=""

    if [[ "$COLLECT_NSYS" == "true" ]]; then
      NSYS_FILE="nsys_vllm_${MODEL//\//_}_conc${CONCURRENCY}"
      echo "▶️ Starting Nsight Systems capture: $NSYS_FILE"
      NSYS_START_ARGS=$(python3 -m yq -r ".models[$i].profiling.nsys_start_args // \"\"" < "$CONFIG_FILE")
      nsys start ${NSYS_START_ARGS}   --output "$NSYS_FILE" 
    fi

    if ! vllm bench serve \
      --base-url "http://localhost:$PORT" \
      --model "$MODEL" \
      --dataset-name random \
      --random-input-len $INPUT_LEN \
      --random-output-len $OUTPUT_LEN \
      --max-concurrency $CONCURRENCY \
      --num-prompts $NUM_PROMPTS \
      --seed $(date +%s) \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 90,95,99 \
      --ignore-eos \
      --save-result \
      --result-filename "$OUTFILE" \
      --append-result; then
        STATUS="failed"
    fi

    if [[ "$COLLECT_NSYS" == "true" ]]; then
      echo "🛑 Stopping Nsight Systems capture..."
      nsys stop
      echo "✅ Nsight report written to ${NSYS_FILE}.*"
    fi

    END_TS=$(date +%s)
    RUNTIME=$((END_TS - START_TS))

    echo "$MODEL,$CONCURRENCY,$NUM_PROMPTS,$STATUS,$RUNTIME,$OUTFILE,$LOGFILE,$NSYS_FILE" >> "$SUMMARY_CSV"
  done

  # Kill server
  if [[ "$COLLECT_NSYS" == "true" ]]; then
	pkill -TERM -P ${VLLM_PID} || true   # kill children first
	kill -TERM ${VLLM_PID} || true       # kill wrapper
	sleep 5
  else
  	echo "🛑 Killing vLLM server for $MODEL"
  	kill -9 -$PGID || true
  	sleep 30
  fi
done

echo "🎉 All benchmarking completed."
echo "📊 Summary written to $SUMMARY_CSV"

