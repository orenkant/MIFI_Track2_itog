"""Загрузка, очистка и разбиение данных на train/test."""

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAW_TARGET = "default.payment.next.month"
TARGET = "target"

PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

FEATURE_COLS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + PAY_STATUS_COLS
    + BILL_COLS
    + PAY_AMT_COLS
)


def load_raw(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()
    df = df.rename(columns={RAW_TARGET: TARGET})
    df = df.drop(columns=["ID"], errors="ignore")

    # коды 0, 5, 6 не описаны в документации -> относим к "прочее" (4)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    # код 0 не описан -> "прочее" (3)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_data(df, test_size=0.2, random_state=42):
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)["data"]

    raw = load_raw(params["raw_path"])
    df = clean_data(raw)
    print(f"Сырые данные: {raw.shape}, после очистки: {df.shape}")

    train, test = split_data(df, params["test_size"], params["random_state"])

    out_dir = Path(params["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    print(
        f"train: {train.shape}, доля дефолтов {train[TARGET].mean():.3f}; "
        f"test: {test.shape}, доля дефолтов {test[TARGET].mean():.3f}"
    )


if __name__ == "__main__":
    main()
