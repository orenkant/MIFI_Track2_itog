import pandas as pd
import pytest

from src.data.make_dataset import clean_data
from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_features,
)


def make_client(**overrides):
    row = {
        "LIMIT_BAL": 100000,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 30,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
    }
    for i in range(1, 7):
        row[f"BILL_AMT{i}"] = 10000
        row[f"PAY_AMT{i}"] = 1000
    row.update(overrides)
    return pd.DataFrame([row])


def test_all_feature_columns_created(raw_sample):
    df = add_features(clean_data(raw_sample))
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        assert col in df.columns
    assert not df[NUMERIC_FEATURES].isnull().any().any()


def test_delay_features():
    df = add_features(make_client(PAY_0=2, PAY_2=1, PAY_3=-1, PAY_4=-2))
    assert df.loc[0, "MONTHS_DELAYED"] == 2
    assert df.loc[0, "MAX_DELAY"] == 2


def test_max_delay_not_negative():
    df = add_features(make_client(PAY_0=-2, PAY_2=-1))
    assert df.loc[0, "MAX_DELAY"] == 0


def test_utilization_and_ratio():
    df = add_features(make_client())
    assert df.loc[0, "UTILIZATION"] == pytest.approx(0.1)
    assert df.loc[0, "PAY_RATIO"] == pytest.approx(0.1)


def test_pay_ratio_without_debt():
    # отрицательный счёт (переплата) не должен давать деление на ноль или inf
    bills = {f"BILL_AMT{i}": 0 for i in range(1, 7)}
    bills["BILL_AMT1"] = -500
    df = add_features(make_client(**bills))
    assert df.loc[0, "PAY_RATIO"] == 1.0


@pytest.mark.parametrize(
    "age, expected", [(21, "<25"), (25, "25-34"), (44, "35-44"), (79, "65+")]
)
def test_age_bins(age, expected):
    assert add_features(make_client(AGE=age)).loc[0, "AGE_BIN"] == expected


def test_add_features_does_not_modify_input():
    df = make_client()
    cols = list(df.columns)
    add_features(df)
    assert list(df.columns) == cols
