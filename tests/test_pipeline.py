import pytest

from src.data.columns import FEATURE_COLS, TARGET
from src.data.make_dataset import clean_data
from src.models.pipeline import create_classifier, create_pipeline


@pytest.fixture
def train_data(raw_sample):
    df = clean_data(raw_sample)
    return df[FEATURE_COLS], df[TARGET]


@pytest.mark.parametrize(
    "model_type, params",
    [
        ("logreg", {}),
        ("random_forest", {"n_estimators": 10}),
        ("gradient_boosting", {"n_estimators": 10}),
        ("hist_gradient_boosting", {"max_iter": 10}),
    ],
)
def test_pipeline_predicts_probabilities(train_data, model_type, params):
    X, y = train_data
    pipe = create_pipeline(create_classifier(model_type, params)).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_without_features(train_data):
    X, y = train_data
    pipe = create_pipeline(create_classifier("logreg"), use_features=False).fit(X, y)
    assert "features" not in pipe.named_steps
    assert pipe.predict(X).shape == (len(X),)


def test_pipeline_handles_unknown_category(train_data):
    X, y = train_data
    pipe = create_pipeline(create_classifier("logreg")).fit(X, y)
    new = X.head(1).copy()
    new["EDUCATION"] = 99
    assert pipe.predict_proba(new).shape == (1, 2)


def test_pipeline_is_reproducible(train_data):
    X, y = train_data
    params = {"n_estimators": 10}
    p1 = create_pipeline(create_classifier("random_forest", params)).fit(X, y)
    p2 = create_pipeline(create_classifier("random_forest", params)).fit(X, y)
    assert (p1.predict_proba(X) == p2.predict_proba(X)).all()


def test_unknown_model_type():
    with pytest.raises(ValueError):
        create_classifier("xgboost")
