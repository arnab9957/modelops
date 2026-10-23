from pathlib import Path

import joblib
import sklearn

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"

RANDOM_STATE = 42
MINIMUM_ACCURACY = 0.90


def train_model() -> dict:
    # Load the dataset.
    dataset = load_iris()

    X = dataset.data
    y = dataset.target

    # Keep the test set separate from training.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Preprocessing and model are packaged together.
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

    # Train the model.
    model.fit(X_train, y_train)

    # Evaluate on held-out data.
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))

    print(f"Test accuracy: {accuracy:.4f}")

    # Enforce the initial model-quality gate.
    if accuracy < MINIMUM_ACCURACY:
        raise RuntimeError(
            f"Model rejected: accuracy {accuracy:.4f} "
            f"is below the required {MINIMUM_ACCURACY:.2f}"
        )

    # Save the model and its feature/class metadata.
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

    return {
        "accuracy": accuracy,
        "model_path": str(MODEL_PATH),
    }


if __name__ == "__main__":
    train_model()
