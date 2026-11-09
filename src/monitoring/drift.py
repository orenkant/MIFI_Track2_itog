"""Расчёт Population Stability Index (PSI) для признаков и предсказаний."""

import numpy as np
import pandas as pd

# общепринятые пороги PSI в кредитном скоринге
PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25

# признак с небольшим числом значений (например, PAY_0) считаю дискретным
MAX_DISCRETE_VALUES = 20


def _shares(counts, eps):
    shares = counts / counts.sum()
    # пустой бин дал бы log(0), поэтому заменяю нули маленьким числом
    return np.clip(shares, eps, None)


def psi(reference, current, bins=10, eps=1e-4):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    values = np.unique(reference)
    if len(values) <= MAX_DISCRETE_VALUES:
        categories = np.union1d(values, np.unique(current))
        ref_counts = np.array([(reference == v).sum() for v in categories])
        cur_counts = np.array([(current == v).sum() for v in categories])
    else:
        # границы бинов по квантилям эталонной выборки
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts = np.histogram(reference, edges)[0]
        cur_counts = np.histogram(current, edges)[0]

    ref_share = _shares(ref_counts, eps)
    cur_share = _shares(cur_counts, eps)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_status(value):
    if value < PSI_MODERATE:
        return "stable"
    if value < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def psi_report(reference, current, columns, bins=10):
    rows = []
    for col in columns:
        value = psi(reference[col], current[col], bins=bins)
        rows.append(
            {"feature": col, "psi": round(value, 4), "status": psi_status(value)}
        )
    return pd.DataFrame(rows)
