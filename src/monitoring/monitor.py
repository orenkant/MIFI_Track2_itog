"""Имитация потока новых клиентов в API и расчёт дрифта (PSI) относительно train."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

from src.data.columns import FEATURE_COLS, TARGET
from src.data.validation import validate_dataframe
from src.monitoring.drift import psi, psi_report, psi_status

KEY_FEATURES = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "BILL_AMT1", "PAY_AMT1"]
STATUS_ORDER = ["stable", "moderate", "significant"]


def simulate_drift(df, n, seed=42):
    """Сдвиг состава клиентов, чтобы проверить, что мониторинг его замечает.

    Строки остаются реальными клиентами из теста, меняется только их доля:
    клиентов с просрочкой в последнем месяце становится половина (в train ~23%).
    """
    risky = df[df["PAY_0"] >= 1]
    other = df[df["PAY_0"] < 1]
    n_risky = n // 2
    sample = pd.concat(
        [
            risky.sample(n=n_risky, random_state=seed),
            other.sample(n=n - n_risky, random_state=seed),
        ]
    )
    # и банк снизил лимиты на 30%
    sample["LIMIT_BAL"] = sample["LIMIT_BAL"] * 0.7
    return sample.sample(frac=1, random_state=seed)


def score_via_api(df, api_url, batch_size=500):
    probas, classes = [], []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size][FEATURE_COLS]
        resp = requests.post(
            f"{api_url}/predict_batch", json=chunk.to_dict(orient="records"), timeout=60
        )
        resp.raise_for_status()
        for item in resp.json():
            probas.append(item["default_probability"])
            classes.append(item["default_prediction"])
    return np.array(probas), np.array(classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--drift", action="store_true", help="сдвинуть данные")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="reports/drift_report.json")
    parser.add_argument(
        "--fail-on-drift", action="store_true", help="код 1 при сильном дрифте"
    )
    args = parser.parse_args()

    try:
        health = requests.get(f"{args.api_url}/health", timeout=5).json()
    except requests.ConnectionError:
        sys.exit(f"API недоступен по адресу {args.api_url}, запустите контейнер")
    print(f"API: модель {health['model_name']}, порог {health['threshold']:.3f}")

    reference = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    if args.drift:
        current = simulate_drift(test, args.n_samples, args.seed)
    else:
        current = test.sample(n=args.n_samples, random_state=args.seed)

    # проверка качества данных перед отправкой в модель
    ok, failed = validate_dataframe(
        current[FEATURE_COLS], with_target=False, with_id=False
    )
    if not ok:
        print("Новые данные не прошли валидацию, в модель не отправляю:")
        for item in failed:
            print(f"  - {item['expectation']} {item['kwargs']}")
        sys.exit(1)

    ref_proba, ref_class = score_via_api(reference, args.api_url)
    cur_proba, cur_class = score_via_api(current, args.api_url)

    features = psi_report(reference, current, KEY_FEATURES)
    score_psi = psi(ref_proba, cur_proba)
    overall = max(
        [psi_status(score_psi)] + features["status"].tolist(), key=STATUS_ORDER.index
    )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_reference": len(reference),
        "n_current": len(current),
        "drift_simulated": args.drift,
        "score_psi": round(score_psi, 4),
        "score_status": psi_status(score_psi),
        "features": features.to_dict(orient="records"),
        "overall_status": overall,
        "predicted_default_rate": {
            "reference": round(float(ref_class.mean()), 4),
            "current": round(float(cur_class.mean()), 4),
        },
        # в тестовой выборке есть разметка, поэтому можно посмотреть и прямые метрики
        "actual_default_rate": {
            "reference": round(float(reference[TARGET].mean()), 4),
            "current": round(float(current[TARGET].mean()), 4),
        },
        "current_roc_auc": round(roc_auc_score(current[TARGET], cur_proba), 4),
    }

    print(f"\nPSI по признакам (train: {len(reference)}, новые: {len(current)}):")
    print(features.to_string(index=False))
    print(f"\nPSI вероятностей дефолта: {score_psi:.4f} ({psi_status(score_psi)})")
    rate, actual = report["predicted_default_rate"], report["actual_default_rate"]
    print(f"Доля предсказанных дефолтов: {rate['reference']} -> {rate['current']}")
    print(f"Фактическая доля дефолтов: {actual['reference']} -> {actual['current']}")
    print(f"ROC-AUC на новых данных: {report['current_roc_auc']}")
    print(f"Итог: {overall}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Отчёт сохранён в {args.report}")

    if args.fail_on_drift and overall == "significant":
        sys.exit(1)


if __name__ == "__main__":
    main()
