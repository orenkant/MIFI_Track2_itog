"""Инференс: загрузка модели и предсказание с порогом из обучения."""

import argparse
import json

import joblib
import pandas as pd

from src.data.columns import FEATURE_COLS


def load_model(model_path="models/model.pkl", meta_path="models/model_meta.json"):
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta


def predict(model, df, threshold):
    """Возвращает (классы, вероятности дефолта)."""
    proba = model.predict_proba(df[FEATURE_COLS])[:, 1]
    classes = (proba >= threshold).astype(int)
    return classes, proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="CSV с признаками клиентов")
    parser.add_argument("output", help="куда сохранить предсказания")
    parser.add_argument("--model", default="models/model.pkl")
    parser.add_argument("--meta", default="models/model_meta.json")
    args = parser.parse_args()

    model, meta = load_model(args.model, args.meta)
    df = pd.read_csv(args.input)
    classes, proba = predict(model, df, meta["threshold"])

    out = df.copy()
    out["default_probability"] = proba
    out["default_prediction"] = classes
    out.to_csv(args.output, index=False)
    print(f"Сохранено {len(out)} предсказаний в {args.output}")


if __name__ == "__main__":
    main()
