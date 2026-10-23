from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_model_info():
    with TestClient(app) as client:
        response = client.get("/model/info")

        assert response.status_code == 200

        data = response.json()

        assert data["model_type"] == "LogisticRegression"
        assert len(data["features"]) == 4
        assert len(data["classes"]) == 3


def test_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["prediction"] in [
            "setosa",
            "versicolor",
            "virginica",
        ]
        assert 0.0 <= data["confidence"] <= 1.0


def test_invalid_prediction_input():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "sepal_length": -1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )

        assert response.status_code == 422
