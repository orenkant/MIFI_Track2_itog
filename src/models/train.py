"""Обучение моделей с подбором гиперпараметров и логированием в MLflow."""

import argparse
import json
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

from src.data.columns import FEATURE_COLS, TARGET
from src.models.metrics import compute_metrics, find_best_threshold
from src.models.pipeline import create_classifier, create_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

# MLflow 3 сохраняет sklearn-модели через skops, свои и sklearn-типы указываю явно
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "src.features.build_features.add_features",
]


def load_split(processed_dir):
    train = pd.read_csv(Path(processed_dir) / "train.csv")
    test = pd.read_csv(Path(processed_dir) / "test.csv")
    return (
        train[FEATURE_COLS],
        train[TARGET],
        test[FEATURE_COLS],
        test[TARGET],
    )


def plot_roc(y_true, y_proba, name, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="случайная модель")
    ax.set_title(f"ROC-кривая на тесте: {name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_confusion(y_true, y_pred, name, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(f"Матрица ошибок: {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_importance(pipeline, name, path, top=20):
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_[0])
    else:
        # у HistGradientBoosting нет встроенной важности признаков
        return False

    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    imp = pd.Series(values, index=names).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.plot(kind="barh", ax=ax)
    ax.set_title(f"Важность признаков: {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return True


def build_search(cfg, cv, common):
    classifier = create_classifier(
        cfg["model"], cfg.get("params"), random_state=common["random_state"]
    )
    pipeline = create_pipeline(classifier, use_features=cfg["use_features"])
    grid = {f"classifier__{k}": v for k, v in cfg["param_grid"].items()}

    if cfg["search"] == "grid":
        return GridSearchCV(
            pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=common["n_jobs"]
        )
    return RandomizedSearchCV(
        pipeline,
        grid,
        n_iter=cfg["n_iter"],
        scoring="roc_auc",
        cv=cv,
        n_jobs=common["n_jobs"],
        random_state=common["random_state"],
    )


def run_experiment(cfg, data, common, tmp_dir):
    X_train, y_train, X_test, y_test = data
    name = cfg["name"]
    cv = StratifiedKFold(
        n_splits=common["cv_folds"], shuffle=True, random_state=common["random_state"]
    )

    with mlflow.start_run(run_name=name) as run:
        start = time.time()
        search = build_search(cfg, cv, common)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        # порог подбираю по out-of-fold предсказаниям на train, тест не трогаю
        oof_proba = cross_val_predict(
            best,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=common["n_jobs"],
        )[:, 1]
        threshold = find_best_threshold(y_train, oof_proba)
        fit_time = time.time() - start

        test_proba = best.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= threshold).astype(int)
        test_metrics = compute_metrics(y_test, test_proba, threshold)

        cv_std = search.cv_results_["std_test_score"][search.best_index_]
        best_params = {
            k.replace("classifier__", ""): v for k, v in search.best_params_.items()
        }

        mlflow.set_tags(
            {"model_type": cfg["model"], "use_features": cfg["use_features"]}
        )
        mlflow.log_params(
            {
                "model_type": cfg["model"],
                "use_features": cfg["use_features"],
                "search": cfg["search"],
                "cv_folds": common["cv_folds"],
                "n_candidates": len(search.cv_results_["params"]),
                "train_rows": len(X_train),
                **(cfg.get("params") or {}),
                **best_params,
            }
        )
        mlflow.log_metrics(
            {
                "cv_roc_auc": search.best_score_,
                "cv_roc_auc_std": cv_std,
                "threshold": threshold,
                "fit_time_sec": fit_time,
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )

        run_dir = tmp_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        plot_roc(y_test, test_proba, name, run_dir / "roc_curve.png")
        plot_confusion(y_test, test_pred, name, run_dir / "confusion_matrix.png")
        plot_importance(best, name, run_dir / "feature_importance.png")
        cv_results = pd.DataFrame(search.cv_results_)[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].sort_values("rank_test_score")
        cv_results.to_csv(run_dir / "cv_results.csv", index=False)
        (run_dir / "classification_report.txt").write_text(
            classification_report(y_test, test_pred, digits=3)
        )
        mlflow.log_artifacts(str(run_dir))

        signature = infer_signature(
            X_test.head(100), best.predict_proba(X_test.head(100))
        )
        model_info = mlflow.sklearn.log_model(
            best,
            name="model",
            signature=signature,
            input_example=X_test.head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    print(
        f"{name:24s} cv_auc={search.best_score_:.4f} "
        f"test_auc={test_metrics['roc_auc']:.4f} f1={test_metrics['f1']:.4f} "
        f"порог={threshold:.3f} ({fit_time:.0f} c)"
    )
    return {
        "name": name,
        "run_id": run.info.run_id,
        "model_uri": model_info.model_uri,
        "model": best,
        "threshold": threshold,
        "best_params": best_params,
        "cv_roc_auc": search.best_score_,
        "cv_roc_auc_std": cv_std,
        "test_metrics": test_metrics,
        "test_proba": test_proba,
    }


def plot_all_roc(results, y_test, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        RocCurveDisplay.from_predictions(y_test, r["test_proba"], name=r["name"], ax=ax)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC-кривые всех моделей на тесте")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_best(best, mlflow_cfg):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best["model"], MODELS_DIR / "model.pkl")

    meta = {
        "model_name": best["name"],
        "mlflow_run_id": best["run_id"],
        "threshold": best["threshold"],
        "best_params": best["best_params"],
        "features": FEATURE_COLS,
    }
    with open(MODELS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    metrics = {
        "model_name": best["name"],
        "cv_roc_auc": best["cv_roc_auc"],
        "threshold": best["threshold"],
        **{f"test_{k}": v for k, v in best["test_metrics"].items()},
    }
    with open("metrics.json", "w") as f:
        json.dump(
            {k: v if isinstance(v, str) else round(v, 4) for k, v in metrics.items()},
            f,
            indent=2,
        )

    # регистрирую лучшую модель и вешаю алиас champion (стадии в MLflow 3 устарели)
    name = mlflow_cfg["registered_model_name"]
    version = mlflow.register_model(best["model_uri"], name)
    mlflow.MlflowClient().set_registered_model_alias(name, "champion", version.version)
    print(f"Зарегистрирована модель {name} v{version.version} (alias champion)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--only", nargs="*", help="запустить только эти эксперименты")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)
    common = params["train"]
    mlflow_cfg = common["mlflow"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    data = load_split(params["data"]["processed_dir"])
    experiments = common["experiments"]
    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        results = [run_experiment(cfg, data, common, Path(tmp)) for cfg in experiments]

    # лучшая модель выбирается по CV на train, а не по тесту
    best = max(results, key=lambda r: r["cv_roc_auc"])
    print(f"Лучшая модель по CV: {best['name']}")

    summary = pd.DataFrame(
        [
            {
                "experiment": r["name"],
                "cv_roc_auc": r["cv_roc_auc"],
                "cv_roc_auc_std": r["cv_roc_auc_std"],
                "threshold": r["threshold"],
                **{f"test_{k}": v for k, v in r["test_metrics"].items()},
            }
            for r in results
        ]
    ).sort_values("cv_roc_auc", ascending=False)
    summary.round(4).to_csv(REPORTS_DIR / "experiments.csv", index=False)

    y_test = data[3]
    best_pred = (best["test_proba"] >= best["threshold"]).astype(int)
    plot_all_roc(results, y_test, FIGURES_DIR / "roc_curves_all.png")
    plot_roc(y_test, best["test_proba"], best["name"], FIGURES_DIR / "roc_curve.png")
    plot_confusion(
        y_test, best_pred, best["name"], FIGURES_DIR / "confusion_matrix.png"
    )
    plot_importance(best["model"], best["name"], FIGURES_DIR / "feature_importance.png")

    save_best(best, mlflow_cfg)


if __name__ == "__main__":
    main()
