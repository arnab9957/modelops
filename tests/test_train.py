import os

from src.train import _enable_mlflow_file_store_for_file_uri


def test_enable_mlflow_file_store_for_file_uri_sets_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)

    _enable_mlflow_file_store_for_file_uri("file:///tmp/mlruns")

    assert os.getenv("MLFLOW_ALLOW_FILE_STORE") == "true"


def test_enable_mlflow_file_store_for_file_uri_preserves_non_file_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)

    _enable_mlflow_file_store_for_file_uri("http://127.0.0.1:5000")

    assert os.getenv("MLFLOW_ALLOW_FILE_STORE") is None
