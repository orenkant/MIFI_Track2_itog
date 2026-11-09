import numpy as np
import pandas as pd
import pytest

from src.data.columns import FEATURE_COLS
from src.data.make_dataset import clean_data
from src.data.validation import validate_dataframe
from src.monitoring.drift import psi, psi_report, psi_status
from src.monitoring.monitor import simulate_drift


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_psi_identical_is_zero(rng):
    x = rng.normal(size=5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_same_distribution_is_stable(rng):
    ref, cur = rng.normal(size=5000), rng.normal(size=2000)
    assert psi_status(psi(ref, cur)) == "stable"


def test_psi_detects_shift(rng):
    ref, cur = rng.normal(size=5000), rng.normal(loc=1.0, size=2000)
    assert psi_status(psi(ref, cur)) == "significant"


def test_psi_discrete_with_new_category():
    ref = np.array([0] * 80 + [1] * 20)
    cur = np.array([0] * 50 + [1] * 20 + [5] * 30)
    value = psi(ref, cur)
    assert np.isfinite(value)
    assert value > 0.25


def test_psi_not_negative(rng):
    for _ in range(5):
        assert psi(rng.normal(size=500), rng.normal(size=500)) >= 0


@pytest.mark.parametrize(
    "value, status", [(0.05, "stable"), (0.1, "moderate"), (0.3, "significant")]
)
def test_psi_status(value, status):
    assert psi_status(value) == status


def test_psi_report(raw_sample):
    df = clean_data(raw_sample)
    report = psi_report(df, df, ["LIMIT_BAL", "PAY_0"])
    assert list(report.columns) == ["feature", "psi", "status"]
    assert (report["status"] == "stable").all()


def test_simulated_drift_is_detected_and_valid(raw_sample):
    df = clean_data(raw_sample)
    shifted = simulate_drift(df, n=300)
    assert len(shifted) == 300
    assert (shifted["PAY_0"] >= 1).mean() == 0.5

    report = psi_report(df, shifted, ["LIMIT_BAL", "PAY_0"]).set_index("feature")
    assert (report["status"] == "significant").all()
    # сдвинутые данные всё ещё проходят валидацию (дрифт, а не ошибка в данных)
    ok, _ = validate_dataframe(
        pd.DataFrame(shifted[FEATURE_COLS]), with_target=False, with_id=False
    )
    assert ok
