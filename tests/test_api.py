import json

import joblib
import pytest
from fastapi.testclient import TestClient

from src.api.app import ClientData, app
from src.data.columns import FEATURE_COLS, TARGET
from src.data.make_dataset import clean_data
from src.models.pipeline import create_classifier, create_pipeline

EXAMPLE = ClientData.model_config["json_schema_extra"]["example"]
THRESHOLD = 0.3


@pytest.fixture
def client(raw_sample, tmp_path, monkeypatch):
    # настоящая модель лежит в DVC, для тестов обучаю маленькую на выборке
    df = clean_data(raw_sample)
    model = create_pipeline(create_classifier("logreg"))
    model.fit(df[FEATURE_COLS], df[TARGET])
    model_path, meta_path = tmp_path / "model.pkl", tmp_path / "meta.json"
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps({"model_name": "test", "threshold": THRESHOLD}))

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("META_PATH", str(meta_path))
    with TestClient(app) as c:
        yield c


def test_root(client):
    assert client.get("/").status_code == 200


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["threshold"] == THRESHOLD


def test_predict(client):
    resp = client.post("/predict", json=EXAMPLE)
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["default_probability"] <= 1
    # класс должен соответствовать порогу из meta
    assert body["default_prediction"] == int(body["default_probability"] >= THRESHOLD)


def test_predict_batch(client, raw_sample):
    rows = raw_sample[FEATURE_COLS].head(5).to_dict(orient="records")
    resp = client.post("/predict_batch", json=rows)
    assert resp.status_code == 200
    assert len(resp.json()) == 5


@pytest.mark.parametrize(
    "field, value", [("AGE", 10), ("SEX", 3), ("PAY_0", 15), ("LIMIT_BAL", -100)]
)
def test_predict_rejects_invalid_values(client, field, value):
    resp = client.post("/predict", json={**EXAMPLE, field: value})
    assert resp.status_code == 422


def test_predict_rejects_missing_field(client):
    data = {k: v for k, v in EXAMPLE.items() if k != "PAY_0"}
    assert client.post("/predict", json=data).status_code == 422


def test_predict_rejects_extra_field(client):
    assert client.post("/predict", json={**EXAMPLE, "ID": 1}).status_code == 422


def test_predict_batch_rejects_empty_list(client):
    assert client.post("/predict_batch", json=[]).status_code == 422
