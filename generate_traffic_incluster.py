import os
import requests
import time
import random
import json
from datetime import datetime

SERVER_URL = "http://idle-risk-exporter.inference.svc.cluster.local:8090/v1/chat/completions"

PROMPTS = [
    "Explain how a hash table works.",
    "Write a short poem about rain.",
    "What causes tides?",
    "Summarize the plot of a mystery novel.",
    "How does TCP handshake work?",
    "What is the difference between a stack and a queue?",
    "Describe how photosynthesis works.",
    "What is a binary search tree?",
    "Explain the CAP theorem.",
    "How does garbage collection work in Python?",
]

REGIMES = {
    "quiet":    {"burst_range": (0, 2), "idle_range": (60, 150)},
    "moderate": {"burst_range": (2, 5), "idle_range": (20, 60)},
    "busy":     {"burst_range": (5, 10), "idle_range": (5, 25)},
}

# Unset, empty, "0", or non-numeric = run forever (old behavior, for manual/interactive use).
# Set via the pod's DURATION_SECONDS env var for timed collection runs.
try:
    DURATION_SECONDS = float(os.environ.get("DURATION_SECONDS", "")) or None
except ValueError:
    DURATION_SECONDS = None


def elapsed(start_time):
    return time.time() - start_time


def time_is_up(start_time):
    return DURATION_SECONDS is not None and elapsed(start_time) >= DURATION_SECONDS


def send_request(prompt, max_tokens=50):
    start = time.time()
    try:
        response = requests.post(
            SERVER_URL,
            headers={"Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
            timeout=30,
        )
        wall_clock = time.time() - start
        data = response.json()
        usage = data.get("usage", {})
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": prompt,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "wall_clock_seconds": wall_clock,
            "status": "success",
        }
    except Exception as e:
        return {"timestamp": datetime.utcnow().isoformat(), "prompt": prompt, "status": "error", "error": str(e)}


def burst_traffic(n_requests, start_time, gap_range=(0.5, 2)):
    for _ in range(n_requests):
        if time_is_up(start_time):
            return
        prompt = random.choice(PROMPTS)
        result = send_request(prompt)
        print(json.dumps(result), flush=True)
        time.sleep(random.uniform(*gap_range))


def idle_gap(gap_range, start_time):
    # Cap the sleep so a long idle_range doesn't overshoot DURATION_SECONDS by minutes.
    sleep_for = random.uniform(*gap_range)
    if DURATION_SECONDS is not None:
        remaining = DURATION_SECONDS - elapsed(start_time)
        sleep_for = max(0, min(sleep_for, remaining))
    time.sleep(sleep_for)


def main():
    start_time = time.time()
    if DURATION_SECONDS is not None:
        print(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "collection_start",
            "duration_seconds": DURATION_SECONDS,
        }), flush=True)

    while not time_is_up(start_time):
        regime_name = random.choice(list(REGIMES.keys()))
        regime = REGIMES[regime_name]
        regime_duration_cycles = random.randint(3, 6)

        print(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "regime_start",
            "regime": regime_name,
            "cycles": regime_duration_cycles,
        }), flush=True)

        for _ in range(regime_duration_cycles):
            if time_is_up(start_time):
                break
            n = random.randint(*regime["burst_range"])
            if n > 0:
                burst_traffic(n_requests=n, start_time=start_time)
            else:
                # Heartbeat so silence never means "unknown" — only "genuinely idle by design"
                print(json.dumps({"timestamp": datetime.utcnow().isoformat(), "event": "idle_cycle_no_requests"}), flush=True)
            idle_gap(regime["idle_range"], start_time)

    print(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "event": "collection_complete",
        "elapsed_seconds": elapsed(start_time),
    }), flush=True)


if __name__ == "__main__":
    main()