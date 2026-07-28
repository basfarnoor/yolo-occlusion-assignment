"""SORT paper reference: Bewley et al., ICIP 2016 (SORT), arxiv.org/abs/1602.00763.

Task 7: aggregate trials.csv into grouped summary statistics (mean, median,
std, sample count -- never hidden) and compute the required cross-method
comparisons.
"""
from __future__ import annotations

import statistics
from collections import defaultdict


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": "", "median": "", "std": "", "n": 0}
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "std": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def group_summary(trials: list[dict], group_keys: list[str]) -> list[dict]:
    """Groups trials by the given keys (any of: method, gap_length, class_name,
    clip) and reports center-error / IoU / coverage / prediction-time stats
    for each group -- every row states its own sample count."""
    groups = defaultdict(list)
    for t in trials:
        key = tuple(t[k] for k in group_keys)
        groups[key].append(t)

    rows = []
    for key, group_trials in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        center_errors = [float(t["center_error_px"]) for t in group_trials if t["has_box"] in (True, "True")]
        ious = [float(t["iou"]) for t in group_trials]
        coverage = sum(1 for t in group_trials if t["has_box"] in (True, "True")) / len(group_trials)
        pred_times = [float(t["prediction_time_ms"]) for t in group_trials]
        iou_ok_rate = sum(1 for t in group_trials if t["iou_at_least_030"] in (True, "True")) / len(group_trials)
        id_cont_rate = sum(1 for t in group_trials if t["id_continuous"] in (True, "True")) / len(group_trials)

        row = dict(zip(group_keys, key))
        row["n_trials"] = len(group_trials)
        row["coverage_rate"] = round(coverage, 3)
        row["iou_at_least_030_rate"] = round(iou_ok_rate, 3)
        row["id_continuity_rate"] = round(id_cont_rate, 3)
        ce_stats = _stats(center_errors)
        row["center_error_mean_px"] = ce_stats["mean"]
        row["center_error_median_px"] = ce_stats["median"]
        row["center_error_std_px"] = ce_stats["std"]
        row["center_error_n"] = ce_stats["n"]
        iou_stats = _stats(ious)
        row["iou_mean"] = iou_stats["mean"]
        row["iou_median"] = iou_stats["median"]
        row["iou_std"] = iou_stats["std"]
        row["iou_n"] = iou_stats["n"]
        t_stats = _stats(pred_times)
        row["prediction_time_ms_mean"] = t_stats["mean"]
        row["prediction_time_ms_n"] = t_stats["n"]
        rows.append(row)
    return rows
