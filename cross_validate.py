import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

FEATURES_FILE = "features.csv"
PREDICTION_HORIZON_WINDOWS = 2

def load_and_prepare():
    df = pd.read_csv(FEATURES_FILE, parse_dates=["timestamp"])
    df["target_will_be_idle"] = df["is_idle_window"].shift(-PREDICTION_HORIZON_WINDOWS)
    df = df.dropna(subset=["target_will_be_idle"])
    df["target_will_be_idle"] = df["target_will_be_idle"].astype(int)

    feature_cols = [
        "request_count", "total_tokens", "tokens_per_second",
        "mean_latency_seconds", "rolling_request_count_mean",
        "rolling_request_count_std", "rolling_tokens_per_second_mean",
    ]
    return df[feature_cols], df["target_will_be_idle"]

def main():
    X, y = load_and_prepare()
    print(f"Total rows: {len(X)}, class balance: {y.value_counts().to_dict()}")

    tscv = TimeSeriesSplit(n_splits=5)
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)

    scores = cross_val_score(model, X, y, cv=tscv, scoring="roc_auc")
    valid_scores = scores[~np.isnan(scores)]
    n_dropped = len(scores) - len(valid_scores)

    print(f"\nROC-AUC across 5 time-ordered folds: {scores}")
    if n_dropped > 0:
        print(f"({n_dropped} fold(s) undefined — test split contained only one class, excluded from mean/std)")
    if len(valid_scores) > 0:
        print(f"Mean (valid folds only): {valid_scores.mean():.3f}  |  Std: {valid_scores.std():.3f}")
        print(f"Range: [{valid_scores.min():.3f}, {valid_scores.max():.3f}]")
    else:
        print("No valid folds to compute mean/std from.")

if __name__ == "__main__":
    main()
