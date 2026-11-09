"""Тесты валидации: на корректных данных проходит, на аномалиях падает."""

import numpy as np
import pytest

from src.data.columns import RAW_TARGET
from src.data.validation import validate_dataframe


def failed_types(failed):
    return {f["expectation"] for f in failed}


def test_valid_sample_passes(raw_sample):
    success, failed = validate_dataframe(raw_sample)
    assert success, failed


def test_fails_on_null(raw_sample):
    df = raw_sample.copy()
    df.loc[0, "LIMIT_BAL"] = np.nan
    success, failed = validate_dataframe(df)
    assert not success
    assert "expect_column_values_to_not_be_null" in failed_types(failed)


def test_fails_on_age_out_of_range(raw_sample):
    df = raw_sample.copy()
    df.loc[0, "AGE"] = 150
    success, failed = validate_dataframe(df)
    assert not success
    assert "expect_column_values_to_be_between" in failed_types(failed)


@pytest.mark.parametrize("col, value", [("SEX", 3), ("EDUCATION", 9), ("PAY_0", 12)])
def test_fails_on_unknown_category(raw_sample, col, value):
    df = raw_sample.copy()
    df.loc[0, col] = value
    success, failed = validate_dataframe(df)
    assert not success
    assert "expect_column_values_to_be_in_set" in failed_types(failed)


def test_fails_on_bad_target(raw_sample):
    df = raw_sample.copy()
    df.loc[0, RAW_TARGET] = 2
    success, _ = validate_dataframe(df)
    assert not success


def test_fails_on_missing_column(raw_sample):
    df = raw_sample.drop(columns=["PAY_0"])
    success, failed = validate_dataframe(df)
    assert not success
    assert "expect_column_to_exist" in failed_types(failed)


def test_fails_on_wrong_type(raw_sample):
    df = raw_sample.copy()
    df["AGE"] = df["AGE"].astype(str)
    success, _ = validate_dataframe(df)
    assert not success


def test_fails_on_duplicate_id(raw_sample):
    df = raw_sample.copy()
    df.loc[1, "ID"] = df.loc[0, "ID"]
    success, failed = validate_dataframe(df)
    assert not success
    assert "expect_column_values_to_be_unique" in failed_types(failed)


def test_new_data_without_target_and_id(raw_sample):
    df = raw_sample.drop(columns=["ID", RAW_TARGET])
    success, failed = validate_dataframe(df, with_target=False, with_id=False)
    assert success, failed
