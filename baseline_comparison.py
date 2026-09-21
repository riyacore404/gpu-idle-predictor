"""
baseline_comparison.py — does the gradient-boosted model actually beat simple
non-ML heuristics on the same prediction task?

Uses the identical time-based train/test split and target construction as
train_model.py, so this is a fair comparison on the same held-out data —
not a separate evaluation with different rules.

Baselines tested:
  1. current_zero:     predict idle if current window's request_count == 0
  2. low_throughput:    predict idle if current window's tokens_per_second
                         is below the median observed in training
  3. rolling_mean_low:  predict idle if rolling_request_count_mean is below
                         the median observed in training
  4. persistence:       predict idle with probability equal to the fraction
                         of idle windows observed in the training set
                         (a "no-signal" baseline — anything at or below this
                         means the model/heuristic isn't learning anything)
"""

import pandas as pd
from sklearn.metrics import roc_auc_score
import joblib

FEATURES_FILE = "features.csv"
MODEL_PATH = "idle_risk_model.joblib"
PREDICTION_HORIZON_WINDOWS = 2


def load_features(path):
    return pd.read_csv(path, parse_dates=["timestamp"])


def build_target(df, horizon=PREDICTION_HORIZON_WINDOWS):
    df = df.copy()
    df["target_will_be_idle"] = df["is_idle_window"].shift(-horizon)
    df = df.dropna(subset=["target_will_be_idle"])
    df["target_will_be_idle"] = df["target_will_be_idle"].astype(int)
    return df


def main():
    print(f"Loading {FEATURES_FILE}...")
    df = load_features(FEATURES_FILE)
    df = build_target(df)

    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    y_test = test["target_will_be_idle"]

    if y_test.nunique() < 2:
        print("Test set is single-class — ROC-AUC undefined for baselines too. "
              "Collect more data or use a different split for this comparison.")
        return

    print(f"Train: {len(train)} rows, Test: {len(test)} rows\n")

    results = {}

    # Baseline 1: current window is already empty
    scores = (test["request_count"] == 0).astype(float)
    results["current_zero (heuristic, no model)"] = roc_auc_score(y_test, scores)

    # Baseline 2: low current throughput
    median_tps = train["tokens_per_second"].median()
    scores = (test["tokens_per_second"] <= median_tps).astype(float)
    results[f"low_throughput (<= train median {median_tps:.3f} tok/s)"] = roc_auc_score(y_test, scores)

    # Baseline 3: low rolling request-count mean
    median_rrm = train["rolling_request_count_mean"].median()
    scores = (test["rolling_request_count_mean"] <= median_rrm).astype(float)
    results[f"rolling_mean_low (<= train median {median_rrm:.3f})"] = roc_auc_score(y_test, scores)

    # Baseline 4: no-signal reference — constant score equal to training idle rate.
    # A constant score gives an undefined/degenerate AUC in sklearn for a
    # single threshold, so report it as the trivial reference point (0.5)
    # rather than computing it directly — any baseline/model should clear 0.5
    # to be doing anything at all.
    idle_rate = train["is_idle_window"].mean()
    print(f"(reference) training idle-window rate: {idle_rate:.3f} — a coin-flip/"
          f"no-signal classifier scores ROC-AUC = 0.500 by definition\n")

    # The actual model
    model = joblib.load(MODEL_PATH)
    feature_cols = [
        "request_count", "total_tokens", "tokens_per_second", "mean_latency_seconds",
        "rolling_request_count_mean", "rolling_request_count_std", "rolling_tokens_per_second_mean",
    ]
    proba = model.predict_proba(test[feature_cols])[:, 1]
    results["gradient_boosted_model (trained)"] = roc_auc_score(y_test, proba)

    print("ROC-AUC on identical held-out test set:")
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name}: {score:.3f}")

    best_baseline = max(v for k, v in results.items() if "gradient_boosted" not in k)
    model_score = results["gradient_boosted_model (trained)"]
    print(f"\nModel vs. best simple baseline: {model_score:.3f} vs {best_baseline:.3f} "
          f"({'+' if model_score > best_baseline else ''}{model_score - best_baseline:.3f})")


if __name__ == "__main__":
    main()