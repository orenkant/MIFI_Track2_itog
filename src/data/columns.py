"""Названия колонок датасета. Модуль без тяжёлых зависимостей, нужен и в API."""

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
