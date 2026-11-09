"""Сборка sklearn Pipeline: признаки -> предобработка -> модель."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features.build_features import (
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_features,
)

MODELS = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def create_classifier(model_type, params=None, random_state=42):
    params = dict(params or {})
    if model_type not in MODELS:
        raise ValueError(f"Неизвестная модель: {model_type}")
    params.setdefault("random_state", random_state)
    if model_type == "logreg":
        params.setdefault("max_iter", 1000)
    return MODELS[model_type](**params)


def create_preprocessor(numeric_features, categorical_features):
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ]
    )


def create_pipeline(classifier, use_features=True):
    """use_features=False - baseline на исходных признаках без FE."""
    steps = []
    if use_features:
        steps.append(("features", FunctionTransformer(add_features)))
        numeric, categorical = NUMERIC_FEATURES, CATEGORICAL_FEATURES
    else:
        numeric, categorical = BASE_NUMERIC_FEATURES, BASE_CATEGORICAL_FEATURES

    steps.append(("preprocessor", create_preprocessor(numeric, categorical)))
    steps.append(("classifier", classifier))
    return Pipeline(steps)
