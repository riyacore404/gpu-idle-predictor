import requests
import time
import random
import json
from datetime import datetime

SERVER_URL = "http://localhost:8080/v1/chat/completions"
LOG_FILE = "traffic_log.jsonl"

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

# Each regime defines: burst size range, and idle gap range after each burst.
# Regimes PERSIST for several cycles — this is what creates real autocorrelation
# in the traffic pattern, instead of independent random noise every cycle.
REGIMES = {
    "quiet":    {"burst_range": (0, 2), "idle_range": (60, 150)},
    "moderate": {"burst_range": (2, 5), "idle_range": (20, 60)},
    "busy":     {"burst_range": (5, 10), "idle_range": (5, 25)},
}

def send_request(prompt, max_tokens=50):
    start = time.time()
    try:
        response = requests.post(
            SERVER_URL,
            headers={"Content-Type": "application/json"},
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=30,
        )
        elapsed = time.time() - start
        data = response.json()
        usage = data.get("usage", {})
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": prompt,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "wall_clock_seconds": elapsed,
            "status": "success",
        }
    except Exception as e:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": prompt,
            "status": "error",
            "error": str(e),
        }

def burst_traffic(n_requests, gap_range=(0.5, 2)):
    results = []
    for _ in range(n_requests):
        prompt = random.choice(PROMPTS)
        result = send_request(prompt)
        results.append(result)
        time.sleep(random.uniform(*gap_range))
    return results

def idle_gap(gap_range):
    gap = random.uniform(*gap_range)
    print(f"  [idle for {gap:.1f}s]")
    time.sleep(gap)

def main():
    print(f"Starting traffic generation. Logging to {LOG_FILE}")
    print("Press Ctrl+C to stop.\n")

    with open(LOG_FILE, "a") as f:
        request_count = 0
        while True:
            regime_name = random.choice(list(REGIMES.keys()))
            regime = REGIMES[regime_name]
            regime_duration_cycles = random.randint(3, 6)  # regime persists for several cycles

            print(f"=== Entering regime: {regime_name} (for {regime_duration_cycles} cycles) ===")

            for cycle in range(regime_duration_cycles):
                n = random.randint(*regime["burst_range"])

                if n > 0:
                    print(f"  Burst: {n} requests")
                    results = burst_traffic(n_requests=n)
                    for r in results:
                        f.write(json.dumps(r) + "\n")
                        f.flush()
                        request_count += 1
                        if r["status"] == "success":
                            print(f"    -> {r['prompt_tokens']}p / {r['completion_tokens']}c tokens, {r['wall_clock_seconds']:.2f}s")
                        else:
                            print(f"    -> ERROR: {r['error']}")
                else:
                    print(f"  [no requests this cycle]")

                idle_gap(regime["idle_range"])

            print(f"Regime '{regime_name}' finished. Total requests so far: {request_count}\n")

if __name__ == "__main__":
    main()
