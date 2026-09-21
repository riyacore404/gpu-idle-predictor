"""
correlate_real_cpu.py — validates whether traffic-based idle windows actually
correspond to measurably lower REAL CPU utilization, using cAdvisor metrics
(container_cpu_usage_seconds_total) already scraped by kube-prometheus-stack.

This directly addresses the gap both engineering audits flagged: the model
predicts request inactivity, not observed resource idleness. cAdvisor gives
genuine (not simulated) per-container CPU usage, so this is a real, if
partial, validation of whether the traffic signal actually tracks resource
idleness for this workload — CPU is a proxy for "the container is doing
real work," which on a CPU-bound llama.cpp inference server is a meaningful
(if imperfect, and not GPU-specific) idleness signal.

Usage:
    python3 correlate_real_cpu.py features.csv

Requires: a Prometheus port-forward already running on localhost:9090
    kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090 &
"""

import sys
import requests
import pandas as pd
from scipy import stats

PROMETHEUS_URL = "http://localhost:9090"
NAMESPACE = "inference"
CONTAINER = "llama-server"
STEP_SECONDS = 30  # match extract_features.py's WINDOW_SECONDS


def query_range_cpu(start_ts, end_ts):
    """
    Query real per-second CPU usage (as a fraction of one core) for the
    llama-cpu container over [start_ts, end_ts], bucketed at STEP_SECONDS —
    same window size as the traffic-based features, so they can be joined.
    """
    query = (
        f'rate(container_cpu_usage_seconds_total'
        f'{{namespace="{NAMESPACE}", container="{CONTAINER}"}}[1m])'
    )
    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={
            "query": query,
            "start": start_ts.timestamp(),
            "end": end_ts.timestamp(),
            "step": f"{STEP_SECONDS}s",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if data["status"] != "success" or not data["data"]["result"]:
        raise RuntimeError(
            f"No CPU data returned for container='{CONTAINER}' in namespace='{NAMESPACE}'. "
            f"Check the query manually at {PROMETHEUS_URL}/graph , and confirm the traffic "
            f"window overlaps a period Prometheus actually scraped (data doesn't survive a "
            f"cluster rebuild)."
        )

    values = data["data"]["result"][0]["values"]
    cpu_df = pd.DataFrame(values, columns=["timestamp", "cpu_fraction"])
    cpu_df["timestamp"] = pd.to_datetime(cpu_df["timestamp"], unit="s")
    cpu_df["cpu_fraction"] = cpu_df["cpu_fraction"].astype(float)
    return cpu_df


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 correlate_real_cpu.py features.csv")
        sys.exit(1)

    features_path = sys.argv[1]
    print(f"Loading {features_path}...")
    features = pd.read_csv(features_path, parse_dates=["timestamp"])
    print(f"Loaded {len(features)} feature windows "
          f"({features['timestamp'].min()} to {features['timestamp'].max()})")

    start_ts = features["timestamp"].min()
    end_ts = features["timestamp"].max() + pd.Timedelta(seconds=STEP_SECONDS)

    print(f"\nQuerying real cAdvisor CPU usage for container='{CONTAINER}' "
          f"in namespace='{NAMESPACE}'...")
    cpu_df = query_range_cpu(start_ts, end_ts)
    print(f"Got {len(cpu_df)} real CPU data points")

    # Join on nearest timestamp (Prometheus scrape times won't exactly match
    # the traffic log's window boundaries) within one window's tolerance.
    merged = pd.merge_asof(
        features.sort_values("timestamp"),
        cpu_df.sort_values("timestamp"),
        on="timestamp",
        tolerance=pd.Timedelta(seconds=STEP_SECONDS),
        direction="nearest",
    )
    merged = merged.dropna(subset=["cpu_fraction"])
    print(f"Joined {len(merged)} windows with matching real CPU data\n")

    if merged["is_idle_window"].nunique() < 2:
        print("All windows are the same idle/busy class in this run — "
              "can't compare groups. Collect a run that spans both a busy "
              "and a quiet stretch.")
        return

    idle_cpu = merged.loc[merged["is_idle_window"] == 1, "cpu_fraction"]
    busy_cpu = merged.loc[merged["is_idle_window"] == 0, "cpu_fraction"]

    print("Real CPU usage (fraction of one core) by traffic-based idle label:")
    print(f"  Traffic-idle windows   (n={len(idle_cpu)}): mean={idle_cpu.mean():.4f}, "
          f"median={idle_cpu.median():.4f}")
    print(f"  Traffic-active windows (n={len(busy_cpu)}): mean={busy_cpu.mean():.4f}, "
          f"median={busy_cpu.median():.4f}")

    if len(idle_cpu) >= 2 and len(busy_cpu) >= 2:
        t_stat, p_value = stats.mannwhitneyu(busy_cpu, idle_cpu, alternative="greater")
        print(f"\nMann-Whitney U test (H1: active-window CPU > idle-window CPU):")
        print(f"  U={t_stat:.2f}, p={p_value:.4f}")
        if p_value < 0.05:
            print("  -> Statistically significant: traffic-based idle labels DO correspond "
                  "to measurably lower real CPU usage in this run.")
        else:
            print("  -> NOT statistically significant at this sample size — traffic-based "
                  "idle labels do not clearly track real CPU usage here. Report this "
                  "honestly; it's a real finding either way.")

    correlation = merged["cpu_fraction"].corr(merged["total_tokens"])
    print(f"\nCorrelation (total_tokens vs real cpu_fraction): {correlation:.3f}")


if __name__ == "__main__":
    main()
