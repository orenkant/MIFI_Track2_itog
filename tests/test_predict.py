import json

import joblib

from src.data.columns import FEATURE_COLS, TARGET
from src.data.make_dataset import clean_data
from src.models.pipeline import create_classifier, create_pipeline
from src.models.predict import load_model, predict


def test_load_and_predict(raw_sample, tmp_path):
    df = clean_data(raw_sample)
    model = create_pipeline(create_classifier("logreg"))
    model.fit(df[FEATURE_COLS], df[TARGET])
    joblib.dump(model, tmp_path / "model.pkl")
    (tmp_path / "meta.json").write_text(json.dumps({"threshold": 0.4}))

    loaded, meta = load_model(tmp_path / "model.pkl", tmp_path / "meta.json")
    classes, proba = predict(loaded, df, meta["threshold"])

    assert len(classes) == len(df)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert ((proba >= 0.4) == classes.astype(bool)).all()


def test_predict_ignores_column_order(raw_sample, tmp_path):
    df = clean_data(raw_sample)
    model = create_pipeline(create_classifier("logreg"))
    model.fit(df[FEATURE_COLS], df[TARGET])
    shuffled = df[FEATURE_COLS[::-1]]
    _, p1 = predict(model, df, 0.5)
    _, p2 = predict(model, shuffled, 0.5)
    assert (p1 == p2).all()
