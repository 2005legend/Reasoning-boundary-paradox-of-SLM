#!/usr/bin/env bash
# Run a queue of experiments unattended on one GPU. Safe to re-run: finished lines are skipped and
# an interrupted training run resumes from its last checkpoint.
#
#   bash scripts/run_queue.sh queues/main_0.5B.txt --s3 s3://BUCKET/heg-grpo \
#        --train-args "--steps 400 --lr 1e-5" --usd-per-hour 1.006 --max-hours 120 --shutdown-when-done
#
# Queue lines:   calibrate <size> [args]             choose MEG's partitioner (scripts/calibrate_meg.py)
#                base <size>                         base-model final eval
#                train <gate> <seed> <size> [args]   train + final eval (line args override --train-args)
set -uo pipefail

usage() { sed -n '2,10p' "$0"; }

QUEUE=""; S3=""; USD="1.006"; SHUTDOWN=0; MAX_HOURS=""; TRAIN_ARGS=""
BENCHES="${EVAL_BENCHMARKS:-math500,gsm8k}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --s3) S3="$2"; shift 2 ;;
    --usd-per-hour) USD="$2"; shift 2 ;;
    --shutdown-when-done) SHUTDOWN=1; shift ;;
    --max-hours) MAX_HOURS="$2"; shift 2 ;;
    --benchmarks) BENCHES="$2"; shift 2 ;;
    --train-args) TRAIN_ARGS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) QUEUE="$1"; shift ;;
  esac
done
[[ -f "$QUEUE" ]] || { usage; exit 1; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
QUEUE="$(cd "$(dirname "$QUEUE")" && pwd)/$(basename "$QUEUE")"
# setup_instance.sh already downloaded every model and dataset; running offline means a slow or
# failing Hugging Face Hub can never stall a paid run. Set HF_ONLINE=1 to allow downloads.
if [[ "${HF_ONLINE:-0}" != "1" ]]; then
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
# The pilot's OOM showed 3.5 GiB reserved-but-unusable memory: let the allocator grow segments instead.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HOME/artifacts}"
ART="$ARTIFACT_ROOT"
mkdir -p "$ART/queue_state" "$ART/logs"
STATE="$ART/queue_state/$(basename "$QUEUE").done"
touch "$STATE"
S3_ARGS=(); [[ -n "$S3" ]] && S3_ARGS=(--s3_uri "${S3%/}")
read -r -a GLOBAL_TRAIN <<< "$TRAIN_ARGS"
# shellcheck disable=SC1091
source "${VENV:-$HOME/venv}/bin/activate"
cd "$REPO"

if [[ -n "$MAX_HOURS" ]]; then
  mins=$(awk "BEGIN{print int($MAX_HOURS*60)}")
  sudo shutdown -h "+$mins" "run_queue safety limit (${MAX_HOURS}h)"
  echo "Safety shutdown armed in ${MAX_HOURS}h. Cancel with: sudo shutdown -c"
fi

trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; echo "${s%"${s##*[![:space:]]}"}"; }

fails=0
while IFS= read -r raw || [[ -n "$raw" ]]; do
  line="$(trim "${raw%%#*}")"
  [[ -z "$line" ]] && continue
  if grep -qxF "$line" "$STATE"; then echo "skip (done): $line"; continue; fi
  read -r -a f <<< "$line"
  logf="$ART/logs/$(echo "$line" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-120).log"
  echo "=== $(date '+%F %T') START: $line"
  echo "    log: $logf"
  case "${f[0]}" in
    calibrate)
      python scripts/calibrate_meg.py --model_size "${f[1]}" "${f[@]:2}" </dev/null >>"$logf" 2>&1 ;;
    base)
      python evaluate.py --base --model_size "${f[1]}" --benchmarks "$BENCHES" "${S3_ARGS[@]}" \
        </dev/null >>"$logf" 2>&1 ;;
    train)
      python train.py --gate "${f[1]}" --seed "${f[2]}" --model_size "${f[3]}" \
        --final_eval "$BENCHES" --usd_per_hour "$USD" "${S3_ARGS[@]}" "${GLOBAL_TRAIN[@]}" "${f[@]:4}" \
        </dev/null >>"$logf" 2>&1 ;;
    *)
      echo "bad queue line: $line" >>"$logf"; false ;;
  esac
  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "$line" >> "$STATE"; fails=0
    echo "=== $(date '+%F %T') DONE: $line"
  else
    fails=$((fails + 1))
    echo "=== $(date '+%F %T') FAILED (exit $rc): $line"
    tail -n 25 "$logf"
    if [[ $fails -ge 2 ]]; then
      echo "Two failures in a row: stopping so a systematic problem cannot burn the budget."
      break
    fi
  fi
done < "$QUEUE"

if [[ -n "$S3" ]]; then
  echo "Final S3 sync of $ART -> ${S3%/}"
  aws s3 sync "$ART" "${S3%/}" --only-show-errors || echo "WARNING: final S3 sync failed"
fi
echo "Queue finished: $(wc -l < "$STATE") lines done (state: $STATE)"
if [[ $SHUTDOWN -eq 1 ]]; then
  echo "Shutting down the instance (stops billing for compute)."
  sudo shutdown -h now
fi
