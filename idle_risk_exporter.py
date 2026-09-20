"""
idle_risk_exporter.py — live GPU idle-risk Prometheus exporter.

Design note: llama.cpp's /metrics only exposes cumulative token/time counters,
not a per-request count or per-request latency (unlike vLLM's real Prometheus
metrics, which include e2e_request_latency histograms — the production target
stack would not need this proxy step). To feed the model features computed
the *same way* extract_features.py computed them at training time
(request_count, mean_latency_seconds per 30s window), this exporter sits as a
reverse proxy in front of llama-cpu and observes each request directly,
rather than approximating request-level stats from cumulative counters.

Traffic (real client traffic, or generate_traffic_incluster.py for a demo)
should point at this exporter's PROXY_PORT instead of llama-cpu directly.
Prometheus scrapes METRICS_PORT for the resulting gpu_idle_risk_score gauge.
"""

import os
import time
import threading
import collections
from datetime import datetime

import requests
import joblib
import pandas as pd
from flask import Flask, request, Response
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

# ---- Config ------------------------------------------------------------
LLAMA_CPP_URL = os.environ.get("LLAMA_CPP_URL", "http://llama-cpu.inference.svc.cluster.local:8080")
PORT = int(os.environ.get("PORT", 8090))  # serves both the proxy and /metrics
MODEL_PATH = os.environ.get("MODEL_PATH", "idle_risk_model.joblib")

# Must match extract_features.py / train_model.py exactly, or the live
# features won't match what the model was trained on.
WINDOW_SECONDS = 30
ROLLING_WINDOWS = 3
PREDICTION_HORIZON_WINDOWS = 2  # kept for reference; the model itself already
                                 # encodes this horizon from training

FEATURE_COLS = [
    "request_count",
    "total_tokens",
    "tokens_per_second",
    "mean_latency_seconds",
    "rolling_request_count_mean",
    "rolling_request_count_std",
    "rolling_tokens_per_second_mean",
]

# ---- State ---------------------------------------------------------------
# Keep enough raw request records to cover ROLLING_WINDOWS worth of 30s
# buckets, plus a margin. Thread-safe via a simple lock (low request volume
# expected — this is a proxy for an idle-risk signal, not a hot path).
_lock = threading.Lock()
_records = collections.deque(maxlen=2000)

model = joblib.load(MODEL_PATH)

idle_risk_gauge = Gauge("gpu_idle_risk_score", "Predicted probability the GPU allocation will be idle N windows from now")
request_count_gauge = Gauge("gpu_idle_window_request_count", "Requests observed in the current window")
total_tokens_gauge = Gauge("gpu_idle_window_total_tokens", "Total tokens (prompt+completion) in the current window")
tokens_per_second_gauge = Gauge("gpu_idle_window_tokens_per_second", "Tokens/sec in the current window")
scoring_errors_total = Gauge("gpu_idle_scoring_errors_total", "Count of scoring cycles that failed")

app = Flask(__name__)


# ---- Proxy: forward to llama-cpu, log each request like the training data ---
@app.route("/v1/chat/completions", methods=["POST"])
def proxy_chat_completions():
    start = time.time()
    payload = request.get_json(force=True)
    try:
        resp = requests.post(
            f"{LLAMA_CPP_URL}/v1/chat/completions",
            json=payload,
            timeout=60,
        )
        wall_clock = time.time() - start
        data = resp.json()
        usage = data.get("usage", {})
        record = {
            "timestamp": datetime.utcnow(),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "wall_clock_seconds": wall_clock,
            "status": "success",
        }
        with _lock:
            _records.append(record)
        return Response(resp.content, status=resp.status_code, content_type=resp.headers.get("Content-Type"))
    except Exception as e:
        wall_clock = time.time() - start
        with _lock:
            _records.append({
                "timestamp": datetime.utcnow(),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "wall_clock_seconds": wall_clock,
                "status": "error",
            })
        return Response(str(e), status=502)


# ---- Feature computation — mirrors extract_features.py's logic exactly ----
def compute_current_features():
    """
    Build the same feature row shape as extract_features.py, using only the
    records observed in the last ROLLING_WINDOWS * WINDOW_SECONDS seconds.
    Returns None if there isn't enough history yet for a meaningful reading.
    """
    with _lock:
        records = list(_records)

    if not records:
        return None

    df = pd.DataFrame(records)
    df = df[df["status"] == "success"].copy()
    if df.empty:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    now = pd.Timestamp.utcnow().tz_localize(None)
    lookback_start = now - pd.Timedelta(seconds=WINDOW_SECONDS * (ROLLING_WINDOWS + 1))
    df = df[df["timestamp"] >= lookback_start]
    if df.empty:
        return None

    df = df.set_index("timestamp")
    agg = df.resample(f"{WINDOW_SECONDS}s").agg(
        request_count=("prompt_tokens", "count"),
        total_prompt_tokens=("prompt_tokens", "sum"),
        total_completion_tokens=("completion_tokens", "sum"),
        mean_latency_seconds=("wall_clock_seconds", "mean"),
    )
    agg = agg.fillna(0)
    agg["total_tokens"] = agg["total_prompt_tokens"] + agg["total_completion_tokens"]
    agg["tokens_per_second"] = agg["total_tokens"] / WINDOW_SECONDS

    agg["rolling_request_count_mean"] = agg["request_count"].rolling(ROLLING_WINDOWS, min_periods=1).mean()
    agg["rolling_request_count_std"] = agg["request_count"].rolling(ROLLING_WINDOWS, min_periods=1).std().fillna(0)
    agg["rolling_tokens_per_second_mean"] = agg["tokens_per_second"].rolling(ROLLING_WINDOWS, min_periods=1).mean()

    if agg.empty:
        return None

    # Most recent complete window is the current reading.
    latest = agg.iloc[-1]
    return latest[FEATURE_COLS].to_frame().T


# ---- Background scoring loop ----------------------------------------------
def scoring_loop():
    while True:
        try:
            features = compute_current_features()
            if features is not None:
                request_count_gauge.set(features["request_count"].iloc[0])
                total_tokens_gauge.set(features["total_tokens"].iloc[0])
                tokens_per_second_gauge.set(features["tokens_per_second"].iloc[0])

                proba = model.predict_proba(features)[:, 1][0]
                idle_risk_gauge.set(proba)
        except Exception as e:
            scoring_errors_total.inc()
            print(f"[scoring_loop] error: {e}", flush=True)
        time.sleep(WINDOW_SECONDS)


@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/healthz")
def healthz():
    return {"status": "ok"}, 200


def main():
    t = threading.Thread(target=scoring_loop, daemon=True)
    t.start()
    # Single port serves both the proxy path (/v1/chat/completions) and
    # /metrics — low-traffic exporter, no need for separate listeners.
    app.run(host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
