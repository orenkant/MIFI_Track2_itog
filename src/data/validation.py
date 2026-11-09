"""Валидация входящих данных с помощью Great Expectations (API версии 1.x)."""

import argparse
import json
import sys
from pathlib import Path

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd
from great_expectations.data_context.types.base import ProgressBarsConfig

from src.data.columns import (
    BILL_COLS,
    FEATURE_COLS,
    PAY_AMT_COLS,
    PAY_STATUS_COLS,
    RAW_TARGET,
)

SUITE_NAME = "credit_data_suite"

INT_TYPES = ["int64", "int32", "Int64"]
NUM_TYPES = INT_TYPES + ["float64", "float32", "Float64"]


def build_expectation_suite(with_target=True, with_id=True):
    columns = list(FEATURE_COLS)
    if with_id:
        columns = ["ID"] + columns
    if with_target:
        columns = columns + [RAW_TARGET]

    exp = [gxe.ExpectTableRowCountToBeBetween(min_value=1)]

    # полнота: все колонки на месте и без пропусков
    for col in columns:
        exp.append(gxe.ExpectColumnToExist(column=col))
        exp.append(gxe.ExpectColumnValuesToNotBeNull(column=col))

    # типы
    for col in ["SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY_STATUS_COLS:
        exp.append(
            gxe.ExpectColumnValuesToBeInTypeList(column=col, type_list=INT_TYPES)
        )
    for col in ["LIMIT_BAL"] + BILL_COLS + PAY_AMT_COLS:
        exp.append(
            gxe.ExpectColumnValuesToBeInTypeList(column=col, type_list=NUM_TYPES)
        )

    # валидность: допустимые коды категорий (как в сырых данных, вместе с 0/5/6)
    exp.append(gxe.ExpectColumnValuesToBeInSet(column="SEX", value_set=[1, 2]))
    exp.append(
        gxe.ExpectColumnValuesToBeInSet(column="EDUCATION", value_set=list(range(0, 7)))
    )
    exp.append(
        gxe.ExpectColumnValuesToBeInSet(column="MARRIAGE", value_set=[0, 1, 2, 3])
    )
    for col in PAY_STATUS_COLS:
        exp.append(
            gxe.ExpectColumnValuesToBeInSet(column=col, value_set=list(range(-2, 10)))
        )

    # согласованность: бизнес-диапазоны (взяты с запасом относительно данных)
    exp.append(
        gxe.ExpectColumnValuesToBeBetween(column="AGE", min_value=18, max_value=100)
    )
    exp.append(
        gxe.ExpectColumnValuesToBeBetween(
            column="LIMIT_BAL", min_value=1, max_value=2_000_000
        )
    )
    for col in BILL_COLS:
        exp.append(
            gxe.ExpectColumnValuesToBeBetween(
                column=col, min_value=-1_000_000, max_value=3_000_000
            )
        )
    for col in PAY_AMT_COLS:
        exp.append(
            gxe.ExpectColumnValuesToBeBetween(
                column=col, min_value=0, max_value=3_000_000
            )
        )

    if with_id:
        exp.append(gxe.ExpectColumnValuesToBeUnique(column="ID"))

    if with_target:
        exp.append(gxe.ExpectColumnValuesToBeInSet(column=RAW_TARGET, value_set=[0, 1]))
        # доля дефолтов в обучающих данных около 22%, сильный сдвиг - повод разобраться
        exp.append(
            gxe.ExpectColumnMeanToBeBetween(
                column=RAW_TARGET, min_value=0.05, max_value=0.5
            )
        )

    return gx.ExpectationSuite(name=SUITE_NAME, expectations=exp)


def validate_dataframe(df, with_target=True, with_id=True):
    """Возвращает (успех, список проваленных проверок)."""
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False)

    batch_def = (
        context.data_sources.add_pandas("pandas")
        .add_dataframe_asset("credit_data")
        .add_batch_definition_whole_dataframe("batch")
    )
    suite = context.suites.add(build_expectation_suite(with_target, with_id))
    validation = context.validation_definitions.add(
        gx.ValidationDefinition(name="credit_validation", data=batch_def, suite=suite)
    )
    result = validation.run(batch_parameters={"dataframe": df})

    failed = []
    for r in result.results:
        if not r.success:
            kwargs = {
                k: v for k, v in r.expectation_config.kwargs.items() if k != "batch_id"
            }
            failed.append(
                {
                    "expectation": r.expectation_config.type,
                    "kwargs": kwargs,
                    "unexpected_count": r.result.get("unexpected_count"),
                }
            )
    return result.success, failed


def save_suite(suite, path=f"data/expectations/{SUITE_NAME}.json"):
    # убираю случайные id, чтобы файл не менялся от запуска к запуску
    data = suite.to_json_dict()
    data.pop("id", None)
    for e in data["expectations"]:
        e.pop("id", None)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="CSV с данными в сыром формате")
    parser.add_argument("--no-target", action="store_true")
    parser.add_argument("--no-id", action="store_true")
    parser.add_argument("--report", default="reports/validation_report.json")
    args = parser.parse_args()

    df = pd.read_csv(args.path)
    with_target, with_id = not args.no_target, not args.no_id
    success, failed = validate_dataframe(df, with_target, with_id)

    suite = build_expectation_suite(with_target, with_id)
    if with_target and with_id:
        save_suite(suite)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        report = {
            "file": args.path,
            "rows": len(df),
            "success": success,
            "failed": failed,
        }
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    n_exp = len(suite.expectations)
    if success:
        print(f"Валидация пройдена: {args.path}, проверок {n_exp}")
    else:
        print(f"Валидация НЕ пройдена: {args.path}, провалено {len(failed)} из {n_exp}")
        for item in failed:
            print(f"  - {item['expectation']} {item['kwargs']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
