import json
import pandas as pd
import numpy as np
from datetime import datetime

LOG_FILE = "traffic_log.jsonl"
OUTPUT_FILE = "features.csv"
WINDOW_SECONDS = 30
SESSION_GAP_THRESHOLD_SECONDS = 600  # gaps over 10 min = new session, not real idle signal

def load_traffic_log(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(records)
    df = df[df["status"] == "success"].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def compute_gaps(df):
    df["gap_seconds"] = df["timestamp"].diff().dt.total_seconds()
    return df

def split_into_sessions(df, session_gap_threshold_seconds=SESSION_GAP_THRESHOLD_SECONDS):
    df = df.copy()
    df["session_id"] = (df["gap_seconds"] > session_gap_threshold_seconds).cumsum()
    return df

def bucket_and_aggregate(df, window_seconds=WINDOW_SECONDS):
    df = df.set_index("timestamp")
    agg = df.resample(f"{window_seconds}s").agg(
        request_count=("prompt_tokens", "count"),
        total_prompt_tokens=("prompt_tokens", "sum"),
        total_completion_tokens=("completion_tokens", "sum"),
        mean_latency_seconds=("wall_clock_seconds", "mean"),
        max_latency_seconds=("wall_clock_seconds", "max"),
    )
    agg = agg.fillna(0)
    agg["total_tokens"] = agg["total_prompt_tokens"] + agg["total_completion_tokens"]
    agg["tokens_per_second"] = agg["total_tokens"] / window_seconds
    return agg.reset_index()

def add_burstiness_features(agg, rolling_windows=3):
    agg["rolling_request_count_mean"] = agg["request_count"].rolling(rolling_windows, min_periods=1).mean()
    agg["rolling_request_count_std"] = agg["request_count"].rolling(rolling_windows, min_periods=1).std().fillna(0)
    agg["rolling_tokens_per_second_mean"] = agg["tokens_per_second"].rolling(rolling_windows, min_periods=1).mean()

    agg["is_idle_window"] = (agg["request_count"] == 0).astype(int)
    agg["consecutive_idle_windows"] = (
        agg["is_idle_window"]
        .groupby((agg["is_idle_window"] != agg["is_idle_window"].shift()).cumsum())
        .cumcount()
        + 1
    ) * agg["is_idle_window"]

    return agg

def main():
    print(f"Loading {LOG_FILE}...")
    df = load_traffic_log(LOG_FILE)
    print(f"Loaded {len(df)} successful requests")

    if len(df) == 0:
        print("No successful requests found — check traffic_log.jsonl")
        return

    df = compute_gaps(df)
    df = split_into_sessions(df)
    df["gap_seconds"] = df.groupby("session_id")["timestamp"].diff().dt.total_seconds()
    print(f"Found {df['session_id'].nunique()} distinct sessions")
    print(df.groupby("session_id").size().rename("requests_in_session"))

    session_durations = df.groupby("session_id")["timestamp"].agg(lambda x: (x.max() - x.min()).total_seconds())
    print(f"\nSession durations (seconds):")
    print(session_durations)
    largest_session_id = session_durations.idxmax()
    df = df[df["session_id"] == largest_session_id].copy()
    print(f"\nUsing session {largest_session_id} with {len(df)} requests (longest duration: {session_durations[largest_session_id]:.0f}s)")
    print(f"Session gap stats — mean: {df['gap_seconds'].mean():.2f}s, max: {df['gap_seconds'].max():.2f}s")

    agg = bucket_and_aggregate(df)
    agg = add_burstiness_features(agg)

    agg.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {len(agg)} time-window rows to {OUTPUT_FILE}")
    print(f"Idle windows: {agg['is_idle_window'].sum()} / {len(agg)}")
    print(f"Max consecutive idle windows: {agg['consecutive_idle_windows'].max()}")
    print("\nSample output:")
    print(agg[["timestamp", "request_count", "total_tokens", "is_idle_window", "consecutive_idle_windows"]].head(15))

if __name__ == "__main__":
    main()
