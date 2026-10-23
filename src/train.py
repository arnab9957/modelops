from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import sklearn

from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import os

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"

# TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "ModelOps-Iris-Training"
REGISTERED_MODEL_NAME = "ModelOps-Iris"

RANDOM_STATE = 42
MINIMUM_ACCURACY = 0.90


def train_model() -> dict:
    # --------------------------------------------------
    # 1. Configure MLflow
    # --------------------------------------------------

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # --------------------------------------------------
    # 2. Load and split the dataset
    # --------------------------------------------------

    dataset = load_iris()

    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # --------------------------------------------------
    # 3. Build the model pipeline
    # --------------------------------------------------

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # --------------------------------------------------
    # 4. Train and evaluate
    # --------------------------------------------------

    with mlflow.start_run(run_name="iris-logistic-regression") as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = float(accuracy_score(y_test, predictions))
        precision = float(precision_score(y_test, predictions, average="macro"))
        recall = float(recall_score(y_test, predictions, average="macro"))
        f1 = float(f1_score(y_test, predictions, average="macro"))

        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Precision (macro): {precision:.4f}")
        print(f"Recall (macro): {recall:.4f}")
        print(f"F1 score (macro): {f1:.4f}")

        # --------------------------------------------------
        # 5. Log parameters and evaluation metrics
        # --------------------------------------------------

        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "scaler": "StandardScaler",
                "random_state": RANDOM_STATE,
                "test_size": 0.20,
                "max_iter": 1000,
                "minimum_accuracy": MINIMUM_ACCURACY,
                "sklearn_version": sklearn.__version__,
            }
        )

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1,
            }
        )

        # --------------------------------------------------
        # 6. Apply the model-quality gate
        # --------------------------------------------------

        if accuracy < MINIMUM_ACCURACY:
            mlflow.set_tags(
                {
                    "release_gate": "REJECTED",
                    "rejection_reason": "Accuracy below threshold",
                }
            )

            raise RuntimeError(
                f"Model rejected: accuracy {accuracy:.4f} "
                f"is below the required {MINIMUM_ACCURACY:.2f}"
            )

        mlflow.set_tag("release_gate", "APPROVED")

        # --------------------------------------------------
        # 7. Save the local model artifact
        # --------------------------------------------------

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": model,
            "feature_names": dataset.feature_names,
            "class_names": dataset.target_names.tolist(),
            "accuracy": accuracy,
            "sklearn_version": sklearn.__version__,
        }

        joblib.dump(artifact, MODEL_PATH)

        print(f"Model saved to: {MODEL_PATH}")

        # Also record the existing artifact in MLflow.
        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="local-artifact",
        )

        # --------------------------------------------------
        # 8. Log and register the trained model
        # --------------------------------------------------

        signature = infer_signature(
            X_train,
            model.predict(X_train),
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="iris-model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
            input_example=X_train[:5],
        )

        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Registered model: {REGISTERED_MODEL_NAME}")
        print("Release gate: APPROVED")

        return {
            "run_id": run.info.run_id,
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "model_path": str(MODEL_PATH),
            "release_gate": "APPROVED",
        }


if __name__ == "__main__":
    train_model()
