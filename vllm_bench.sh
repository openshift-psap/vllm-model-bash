CONCURRENCIES=(1 32 128 512 650)

INPUT_LEN=2000
OUTPUT_LEN=200
CC_MULT=10
MODEL=$1
PORT=$2
OUTFILE=${3}.json
BASE_URL=http://localhost:${PORT}


for CONCURRENCY in "${CONCURRENCIES[@]}";
do
    NUM_PROMPTS=$(($CONCURRENCY * $CC_MULT))

    echo ""
    echo "===== RUNNING $MODEL FOR $NUM_PROMPTS PROMPTS WITH CONCURRENCY $CONCURRENCY ====="
    echo ""

    vllm bench serve  \
        --base-url ${BASE_URL} \
        --model ${MODEL} \
        --dataset-name random \
        --random-input-len ${INPUT_LEN} \
        --random-output-len ${OUTPUT_LEN} \
        --max-concurrency ${CONCURRENCY} \
        --num-prompts ${NUM_PROMPTS} \
        --seed $(date +%s) \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 90,95,99 \
        --ignore-eos \
        --save-result \
        --result-filename ${OUTFILE} \
        --append-result

done
