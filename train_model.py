import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
import joblib

FEATURES_FILE = "features.csv"
MODEL_OUTPUT = "idle_risk_model.joblib"

# How far ahead we're predicting: "will this allocation still be idle N windows from now"
# This is the actual prediction task — not just "is it idle right now" (that's just reading
# a metric), but "should we flag this before the next billing cycle confirms it's wasted"
PREDICTION_HORIZON_WINDOWS = 2  # e.g. predict 2 windows (60s) ahead with 30s windows

def load_features(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df

def build_target(df, horizon=PREDICTION_HORIZON_WINDOWS):
    """
    Target: will the allocation be idle `horizon` windows from now?
    This shifts is_idle_window backward so each row's target is a FUTURE idle state,
    not the current one — that's what makes this a genuine prediction task.
    """
    df = df.copy()
    df["target_will_be_idle"] = df["is_idle_window"].shift(-horizon)
    # Drop rows near the end where we don't have a future label yet
    df = df.dropna(subset=["target_will_be_idle"])
    df["target_will_be_idle"] = df["target_will_be_idle"].astype(int)
    return df

def select_features(df):
    """
    Feature columns — everything the model is allowed to see.
    Deliberately excludes is_idle_window and consecutive_idle_windows for the CURRENT
    row's own idle state at prediction time, since in a real deployment we'd know that
    already; the model should learn from the TREND (rolling stats), not just echo the
    present state back as a trivial predictor.
    """
    feature_cols = [
        "request_count",
        "total_tokens",
        "tokens_per_second",
        "mean_latency_seconds",
        "rolling_request_count_mean",
        "rolling_request_count_std",
        "rolling_tokens_per_second_mean",
    ]
    return df[feature_cols], feature_cols

def main():
    print(f"Loading {FEATURES_FILE}...")
    df = load_features(FEATURES_FILE)
    print(f"Loaded {len(df)} rows")

    if len(df) < 20:
        print(f"WARNING: only {len(df)} rows — this is too small for a meaningful train/test split.")
        print("Consider collecting more traffic before training. Proceeding anyway for a smoke test.")

    df = build_target(df)
    print(f"After building target: {len(df)} rows")
    print(f"Target distribution:\n{df['target_will_be_idle'].value_counts()}")

    X, feature_cols = select_features(df)
    y = df["target_will_be_idle"]

    # Time-based split, NOT random — shuffling would leak future information into training,
    # since consecutive rows are correlated (rolling features overlap between neighbors)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\nTrain: {len(X_train)} rows, Test: {len(X_test)} rows")

    if y_train.nunique() < 2:
        print("ERROR: training set has only one class present — cannot train a classifier.")
        print("This will happen with very small/short datasets. Collect more varied traffic.")
        return

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    if len(X_test) > 0 and y_test.nunique() >= 1:
        preds = model.predict(X_test)
        print("\nClassification report (test set):")
        print(classification_report(y_test, preds, zero_division=0))

        if y_test.nunique() == 2:
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
            print(f"ROC-AUC: {auc:.3f}")
    else:
        print("Test set too small or single-class — skipping evaluation metrics.")

    print("\nFeature importances:")
    for name, importance in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {importance:.3f}")

    joblib.dump(model, MODEL_OUTPUT)
    print(f"\nModel saved to {MODEL_OUTPUT}")

if __name__ == "__main__":
    main()
