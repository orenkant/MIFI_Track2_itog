import pandas as pd

from src.data.columns import RAW_TARGET, TARGET
from src.data.make_dataset import clean_data, split_data


def test_clean_data_renames_target_and_drops_id(raw_sample):
    df = clean_data(raw_sample)
    assert TARGET in df.columns
    assert RAW_TARGET not in df.columns
    assert "ID" not in df.columns


def test_clean_data_merges_undocumented_codes():
    df = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "EDUCATION": [0, 5, 6, 2],
            "MARRIAGE": [0, 1, 2, 3],
            RAW_TARGET: [0, 1, 0, 1],
        }
    )
    out = clean_data(df)
    assert out["EDUCATION"].tolist() == [4, 4, 4, 2]
    assert out["MARRIAGE"].tolist() == [3, 1, 2, 3]


def test_clean_data_drops_duplicates_ignoring_id(raw_sample):
    dup = raw_sample.iloc[[0]].copy()
    dup["ID"] = 999999
    df = pd.concat([raw_sample, dup], ignore_index=True)
    assert len(clean_data(df)) == len(clean_data(raw_sample))


def test_split_is_stratified(raw_sample):
    df = clean_data(raw_sample)
    train, test = split_data(df, test_size=0.2, random_state=42)
    assert len(train) + len(test) == len(df)
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.01
