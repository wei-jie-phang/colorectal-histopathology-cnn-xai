import numpy as np
import pandas as pd


# Recursively flatten the dict; keep only scalar int/float leaves
def flatten_numeric(d, parent_key=""):
    items = {}
    for k, v in d.items():
        key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_numeric(v, key))
        elif isinstance(v, bool):
            continue
        elif isinstance(v, (int, float)):
            items[key] = float(v)
    return items


# Aggregate the mean +/- SD for one top-level section across the runs
def aggregate_section(results, section_key):
    flat_per_run = [
        flatten_numeric(r[section_key])
        for r in results
        if section_key in r
    ]

    if not flat_per_run:
        return pd.DataFrame()
    common_keys = set(flat_per_run[0].keys())
    
    for f in flat_per_run[1:]:
        common_keys &= set(f.keys())

    rows = []

    for key in sorted(common_keys):
        values = np.array([f[key] for f in flat_per_run], dtype=float)
        sd = values.std(ddof=1) if len(values) > 1 else 0.0
        rows.append({
            "metric": key,
            "mean": values.mean(),
            "sd": sd,
            "n_runs": len(values),
            "mean_pm_sd": f"{values.mean():.4f} ± {sd:.4f}",
        })

    return pd.DataFrame(rows)