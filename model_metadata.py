"""
Drop-in addition to train_model.py — writes idle_risk_model_metadata.json
alongside idle_risk_model.joblib so the artifact isn't just an opaque
.joblib file with no record of what produced it or what environment it
needs. Addresses the "model trained with sklearn X, loaded with sklearn Y"
silent-mismatch risk.

Usage: add the import and the write_model_metadata() call to train_model.py
right after joblib.dump(model, MODEL_OUTPUT) — see the bottom of this file
for the exact insertion point.
"""

import json
import platform
import sys
from datetime import datetime, timezone

import sklearn
import numpy
import pandas


def write_model_metadata(
    output_path,
    feature_cols,
    prediction_horizon_windows,
    window_seconds,
    rolling_windows,
    train_rows,
    test_rows,
    dataset_file,
):
    metadata = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_schema": {
            "features_in_order": feature_cols,
            "window_seconds": window_seconds,
            "rolling_windows": rolling_windows,
            "prediction_horizon_windows": prediction_horizon_windows,
        },
        "training_data": {
            "source_file": dataset_file,
            "train_rows": train_rows,
            "test_rows": test_rows,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": numpy.__version__,
            "pandas_version": pandas.__version__,
        },
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata written to {output_path}")
    return metadata


# --- Insertion point for train_model.py ---
# After this line in train_model.py's main():
#
#     joblib.dump(model, MODEL_OUTPUT)
#     print(f"\nModel saved to {MODEL_OUTPUT}")
#
# add:
#
#     from model_metadata import write_model_metadata
#     write_model_metadata(
#         output_path="idle_risk_model_metadata.json",
#         feature_cols=feature_cols,
#         prediction_horizon_windows=PREDICTION_HORIZON_WINDOWS,
#         window_seconds=30,          # matches extract_features.py's WINDOW_SECONDS
#         rolling_windows=3,          # matches extract_features.py's rolling_windows arg
#         train_rows=len(X_train),
#         test_rows=len(X_test),
#         dataset_file=FEATURES_FILE,
#     )
#
# The exporter can then optionally validate against this at startup —
# not required for the current prototype, but the metadata now exists
# for that check to be added later without retraining anything.
