#!/usr/bin/env bash
#
# run_collection.sh — hardened in-cluster traffic collection for gpu-idle-predictor
#
# Usage:
#   ./run_collection.sh [duration_seconds] [output_prefix]
#
# Examples:
#   ./run_collection.sh 300              # 5-min smoke test
#   ./run_collection.sh 10800            # 3-hour final collection run
#   ./run_collection.sh 10800 v5_run     # writes v5_run.jsonl / v5_run.stderr.log
#
# Assumes:
#   - generate_traffic_incluster.py reads its run duration from the
#     DURATION_SECONDS env var. If it currently takes duration as a CLI arg
#     or hardcodes it, either change it to read os.environ["DURATION_SECONDS"]
#     or adjust the `kubectl apply` step below to inject it however your
#     script expects.
#   - traffic-generator-pod.yaml's container exits (rather than sleeping
#     forever) once generate_traffic_incluster.py finishes, so pod phase
#     naturally becomes Succeeded.
#   - Namespace: inference. Pod label for llama server: app=llama-cpu.

set -euo pipefail

NAMESPACE="inference"
DURATION_SECONDS="${1:-7200}"
OUTPUT_PREFIX="${2:-traffic_log_incluster}"
LOG_FILE="${OUTPUT_PREFIX}.jsonl"
STDERR_FILE="${OUTPUT_PREFIX}.stderr.log"
POD_NAME="traffic-generator"
CONFIGMAP_NAME="traffic-generator-script"
POD_YAML="traffic-generator-pod.yaml"
MIN_ACCEPTABLE_LINES=50
POLL_INTERVAL=15

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; exit 1; }

# ---- 1. Clean up stale/errored pods before we start ------------------------
log "Cleaning up any stale llama-cpu or traffic-generator pods..."
kubectl delete pod -n "$NAMESPACE" -l app=llama-cpu \
  --field-selector=status.phase!=Running --ignore-not-found --wait=false
kubectl delete pod -n "$NAMESPACE" "$POD_NAME" --ignore-not-found --wait=true 2>/dev/null || true

# ---- 2. Confirm llama-cpu is actually ready before generating traffic ------
log "Waiting for llama-cpu to be Ready..."
kubectl wait --for=condition=Ready pod -l app=llama-cpu -n "$NAMESPACE" --timeout=120s \
  || die "llama-cpu not Ready after 120s — check 'kubectl get pods -n $NAMESPACE'"

# ---- 3. (Re)apply the traffic generator script + pod -----------------------
log "Applying configmap (script) and rendering pod manifest (duration=${DURATION_SECONDS}s)..."
kubectl create configmap "$CONFIGMAP_NAME" \
  -n "$NAMESPACE" \
  --from-file=generate_traffic_incluster.py \
  --dry-run=client -o yaml | kubectl apply -f -

# Env vars must be set at pod CREATION time (kubectl set env on a running pod
# does not propagate to an already-started process) — so render the duration
# into a temp copy of the manifest instead of patching after the fact.
RENDERED_YAML="$(mktemp)"
trap 'rm -f "$RENDERED_YAML"' EXIT
sed "s/DURATION_PLACEHOLDER/${DURATION_SECONDS}/" "$POD_YAML" > "$RENDERED_YAML"
kubectl apply -f "$RENDERED_YAML"
kubectl wait --for=condition=Ready pod/"$POD_NAME" -n "$NAMESPACE" --timeout=60s \
  || die "traffic-generator pod not Ready after 60s"

# ---- 4. Stream logs ----------------------------------------------------------
# Note: kubectl logs merges the pod's stdout+stderr into one stream at the
# container runtime level — a local 2> redirect here only splits kubectl's
# *own* diagnostics, not the pod's. The pip noise that used to land in this
# file is now silenced at the source (--quiet --root-user-action=ignore in
# traffic-generator-pod.yaml), so this should be clean JSONL end to end.
log "Streaming logs to $LOG_FILE..."
kubectl logs -n "$NAMESPACE" "$POD_NAME" -f > "$LOG_FILE" 2> "$STDERR_FILE" &
LOGPID=$!

# ---- 5. Poll for completion instead of trusting a fixed sleep --------------
log "Polling pod phase every ${POLL_INTERVAL}s until Succeeded/Failed (cap: $((DURATION_SECONDS + 600))s)..."
ELAPSED=0
CAP=$((DURATION_SECONDS + 600))  # generous buffer over expected duration
while true; do
  PHASE=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
  if [[ "$PHASE" == "Succeeded" || "$PHASE" == "Failed" ]]; then
    log "Pod reached phase: $PHASE"
    break
  fi
  # Belt-and-suspenders: the script also emits a "collection_complete" sentinel
  # line right before exiting, in case phase reporting lags.
  if grep -q '"event": "collection_complete"' "$LOG_FILE" 2>/dev/null; then
    log "Found collection_complete sentinel in log."
    break
  fi
  if (( ELAPSED >= CAP )); then
    log "WARNING: exceeded expected runtime + buffer (${CAP}s) — pod phase still '$PHASE'. Stopping wait."
    break
  fi
  sleep "$POLL_INTERVAL"
  ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

# Give the log follower a moment to flush the tail, then stop it.
sleep 2
kill "$LOGPID" 2>/dev/null || true
wait "$LOGPID" 2>/dev/null || true

# ---- 6. Sanity-check before deleting anything -------------------------------
LINES=$(wc -l < "$LOG_FILE" | tr -d ' ')
log "Captured $LINES lines in $LOG_FILE"

if (( LINES < MIN_ACCEPTABLE_LINES )); then
  log "WARNING: only $LINES lines captured (threshold: $MIN_ACCEPTABLE_LINES) — NOT deleting pod."
  log "Inspect manually: kubectl logs -n $NAMESPACE $POD_NAME | tail -20"
  exit 1
fi

log "Deleting traffic-generator pod..."
kubectl delete pod -n "$NAMESPACE" "$POD_NAME" --ignore-not-found

log "Done. Output: $LOG_FILE ($LINES lines). Stderr (if any): $STDERR_FILE"
log "Next: python extract_features.py $LOG_FILE  →  train_model.py  →  cross_validate.py"
