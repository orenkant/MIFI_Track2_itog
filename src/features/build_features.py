"""Feature Engineering. Все признаки считаются построчно, без статистик по выборке."""

import numpy as np
import pandas as pd

from src.data.columns import BILL_COLS, PAY_AMT_COLS, PAY_STATUS_COLS

AGE_BINS = [0, 25, 35, 45, 55, 65, np.inf]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]

NEW_NUMERIC_FEATURES = [
    "MONTHS_DELAYED",
    "MAX_DELAY",
    "UTILIZATION",
    "BILL_MEAN",
    "PAY_AMT_MEAN",
    "PAY_RATIO",
]

BASE_NUMERIC_FEATURES = (
    ["LIMIT_BAL", "AGE"] + PAY_STATUS_COLS + BILL_COLS + PAY_AMT_COLS
)
BASE_CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + NEW_NUMERIC_FEATURES
CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES + ["AGE_BIN"]


def add_features(df):
    df = df.copy()
    pay_status = df[PAY_STATUS_COLS]

    df["MONTHS_DELAYED"] = (pay_status > 0).sum(axis=1)
    df["MAX_DELAY"] = pay_status.max(axis=1).clip(lower=0)

    df["UTILIZATION"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
    df["BILL_MEAN"] = df[BILL_COLS].mean(axis=1)
    df["PAY_AMT_MEAN"] = df[PAY_AMT_COLS].mean(axis=1)

    # если суммарный счёт <= 0, долга нет -> считаем, что оплачено полностью
    total_bill = df[BILL_COLS].sum(axis=1)
    total_paid = df[PAY_AMT_COLS].sum(axis=1)
    ratio = total_paid / total_bill.where(total_bill > 0)
    df["PAY_RATIO"] = ratio.fillna(1.0).clip(upper=5.0)

    df["AGE_BIN"] = pd.cut(
        df["AGE"], bins=AGE_BINS, labels=AGE_LABELS, right=False
    ).astype(str)
    return df
