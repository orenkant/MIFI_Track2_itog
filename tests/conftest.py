from pathlib import Path

import pandas as pd
import pytest

SAMPLE_PATH = Path(__file__).parent / "data" / "sample.csv"


@pytest.fixture
def raw_sample():
    # 1000 строк из исходного датасета (стратифицированно по таргету)
    return pd.read_csv(SAMPLE_PATH)
