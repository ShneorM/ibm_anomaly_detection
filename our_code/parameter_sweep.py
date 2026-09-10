"""Run a real parameter sweep using saved reconstruction errors."""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import norm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("replication_config.yaml")
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nab_scoring import calculate_nab_score_with_window_based_tp_fn  # noqa: E402


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def compute_likelihoods(errors: pd.Series, long_window: int, short_window: int) -> np.ndarray:
    """Vectorized equivalent of the likelihood calculation in the source repository."""
    if short_window > long_window:
        raise ValueError("short_window must not be greater than long_window")
    values = errors.to_numpy(dtype=float)
    likelihoods = np.full(len(values), 0.5, dtype=float)
    if len(values) <= long_window:
        return likelihoods

    # These are read-only sliding views, not copies of the full data. Starting
    # the wide view at row 1 matches the source function's first evaluated
    # slice when i == long_window.
    wide = np.lib.stride_tricks.sliding_window_view(values, long_window)[1:]
    narrow = np.lib.stride_tricks.sliding_window_view(values, short_window)[
        long_window - short_window + 1 :
    ]
    wide_mean = np.mean(wide, axis=1)
    wide_std = np.std(wide, axis=1)
    narrow_mean = np.mean(narrow, axis=1)
    z_score = (narrow_mean - wide_mean) / (wide_std + 1e-10)
    # Keep the same survival-function expression as the original implementation.
    # Replacing it with the mathematically equivalent CDF can change values by a
    # few floating-point bits near the 0.9996 decision threshold.
    likelihoods[long_window:] = 0.5 + 0.5 * (1 - norm.sf(z_score))
    return np.nan_to_num(likelihoods, nan=0.5)


def load_test_windows(cfg) -> pd.DataFrame:
    labels = pd.read_csv(resolve_project_path(cfg.paths.labels))
    labels["anomaly_start"] = pd.to_datetime(labels["anomaly_start"], utc=True).dt.tz_localize(None)
    labels["anomaly_end"] = pd.to_datetime(labels["anomaly_end"], utc=True).dt.tz_localize(None)
    labels["anomaly_window_start"] = labels["anomaly_start"] - pd.Timedelta(
        minutes=int(cfg.experiment.anomaly_minutes_before)
    )
    labels["anomaly_window_end"] = labels["anomaly_end"]
    test_start = pd.Timestamp(cfg.experiment.test_start_date)
    end_exclusive = pd.Timestamp(cfg.experiment.end_date) + pd.Timedelta(days=1)
    return labels.loc[
        (labels["anomaly_window_start"] >= test_start)
        & (labels["anomaly_window_end"] <= end_exclusive),
        ["anomaly_window_start", "anomaly_window_end", "anomaly_source"],
    ].copy()


def _scaled_sigmoid(position: float) -> float:
    if position > 3.0:
        return -1.0
    return 2.0 / (1.0 + math.exp(5.0 * position)) - 1.0


def calculate_nab_fast(scoring_frame: pd.DataFrame, test_windows: pd.DataFrame, profile: str) -> tuple:
    """Calculate the repository's NAB score without its verbose row-by-row loop."""
    if profile not in {"standard", "reward_fn"}:
        raise ValueError(f"Unsupported NAB scoring profile: {profile}")
    penalty_fn = 2.0 if profile == "reward_fn" else 1.0
    penalty_fp = 0.11
    index = scoring_frame.index
    predicted = scoring_frame["predicted_anomaly"].to_numpy(dtype=bool)
    true_values = scoring_frame["true_anomaly"].to_numpy(dtype=bool)
    windows = [
        (pd.Timestamp(row.anomaly_window_start), pd.Timestamp(row.anomaly_window_end))
        for row in test_windows.itertuples()
    ]

    # Match calculate_relative_position(): the end of a true block is the first
    # following false sample, not the final true sample.
    changes = np.diff(np.r_[False, true_values, False].astype(int))
    block_starts = np.flatnonzero(changes == 1)
    block_ends_exclusive = np.flatnonzero(changes == -1)
    relative_positions: dict[pd.Timestamp, float] = {}
    for start_pos, end_pos in zip(block_starts, block_ends_exclusive):
        predicted_positions = np.flatnonzero(predicted[start_pos:end_pos])
        if not len(predicted_positions):
            continue
        first_pos = start_pos + int(predicted_positions[0])
        end_time = index[end_pos] if end_pos < len(index) else index[-1]
        duration = (end_time - index[start_pos]).total_seconds()
        relative_positions[index[first_pos]] = (
            -(end_time - index[first_pos]).total_seconds() / duration if duration else 0.0
        )

    score = 0.0
    true_positive_windows = 0
    false_negative_windows = 0
    inside_any_window = np.zeros(len(index), dtype=bool)
    for start, end in windows:
        in_window = (index >= start) & (index <= end)
        inside_any_window |= in_window
        positions = np.flatnonzero(predicted & in_window)
        if len(positions):
            first_time = index[int(positions[0])]
            score += _scaled_sigmoid(relative_positions.get(first_time, 0.0))
            true_positive_windows += 1
        else:
            score -= penalty_fn
            false_negative_windows += 1

    false_positive_alerts = 0
    window_ends = np.array([end.value for _, end in windows], dtype=np.int64)
    predicted_outside = np.flatnonzero(predicted & ~inside_any_window)
    for position in predicted_outside:
        time = index[int(position)]
        previous = int(np.searchsorted(window_ends, time.value, side="left") - 1)
        if previous < 0:
            continue
        start, end = windows[previous]
        width = (end - start).total_seconds()
        relative_position = (time - end).total_seconds() / width
        score += penalty_fp * _scaled_sigmoid(relative_position)
        false_positive_alerts += 1

    baseline = -penalty_fn * len(block_starts)
    perfect = float(len(block_starts))
    normalized = 100.0 * (score - baseline) / (perfect - baseline)
    return score, normalized, false_positive_alerts, false_negative_windows, {
        "tp": true_positive_windows,
        "fn": false_negative_windows,
        "fp": false_positive_alerts,
    }


def evaluate_combination(
    detailed_results: pd.DataFrame,
    test_windows: pd.DataFrame,
    profile: str,
    threshold: float,
    long_window: int,
    short_window: int,
    use_reference_nab: bool = False,
) -> dict:
    likelihoods = compute_likelihoods(detailed_results["reconstruction_error"], long_window, short_window)
    predicted = (likelihoods > threshold).astype(int)
    true_values = detailed_results["true_anomaly"].astype(int).to_numpy()
    scoring_frame = pd.DataFrame(
        {"true_anomaly": true_values, "predicted_anomaly": predicted},
        index=detailed_results.index,
    )
    if use_reference_nab:
        with contextlib.redirect_stdout(io.StringIO()):
            raw_nab, normalized_nab, fp_alerts, fn_windows, counters = (
                calculate_nab_score_with_window_based_tp_fn(
                    scoring_frame.copy(), test_windows.copy(), profile
                )
            )
    else:
        raw_nab, normalized_nab, fp_alerts, fn_windows, counters = calculate_nab_fast(
            scoring_frame, test_windows, profile
        )
    return {
        "threshold": threshold,
        "long_window": long_window,
        "short_window": short_window,
        "precision": precision_score(true_values, predicted, zero_division=0),
        "recall": recall_score(true_values, predicted, zero_division=0),
        "f1_score": f1_score(true_values, predicted, zero_division=0),
        "accuracy": accuracy_score(true_values, predicted),
        "mcc": matthews_corrcoef(true_values, predicted),
        "true_positive_windows": counters["tp"],
        "false_positive_alerts": fp_alerts,
        "false_negative_windows": fn_windows,
        "raw_nab_score": raw_nab,
        "normalized_nab_score": normalized_nab,
    }


def run_sweep(config_path: Path, models: list[str], quick: bool, output: Path | None) -> pd.DataFrame:
    cfg = OmegaConf.load(config_path)
    test_windows = load_test_windows(cfg)
    results_dir = resolve_project_path(cfg.paths.results_dir)
    all_rows = []

    for model in models:
        model = model.upper()
        detail_path = results_dir / f"unweighted__{model}_anomaly_detection_results.csv"
        detailed = pd.read_csv(detail_path, parse_dates=["interval_start"]).set_index("interval_start")
        if quick:
            reported = cfg.reported_results[model]
            combinations = [(reported.threshold, reported.long_window, reported.short_window)]
        else:
            combinations = itertools.product(
                cfg.parameter_grid.thresholds,
                cfg.parameter_grid.long_windows,
                cfg.parameter_grid.short_windows,
            )
        for threshold, long_window, short_window in combinations:
            row = evaluate_combination(
                detailed,
                test_windows,
                cfg.experiment.nab_profile,
                float(threshold),
                int(long_window),
                int(short_window),
            )
            row["model"] = model
            all_rows.append(row)

    result = pd.DataFrame(all_rows)
    result = result.sort_values(["model", "normalized_nab_score"], ascending=[True, False])
    output_path = output or resolve_project_path(cfg.paths.output_dir) / "parameter_sweep.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    minimum_tp = int(cfg.selection.minimum_true_positive_windows)
    summary = {"selection_rule": f"highest NAB score with at least {minimum_tp} detected windows", "models": {}}
    for model in models:
        model_rows = result.loc[result["model"] == model.upper()]
        eligible_rows = model_rows.loc[model_rows["true_positive_windows"] >= minimum_tp]
        summary["models"][model.upper()] = {
            "highest_nab_without_detection_constraint": model_rows.iloc[0].to_dict(),
            "selected_low_fn_configuration": eligible_rows.iloc[0].to_dict(),
        }
    summary_path = output_path.with_name("parameter_sweep_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+", default=["ANN", "GRU"], choices=["ANN", "GRU"])
    parser.add_argument("--quick", action="store_true", help="Check only the reported parameter set")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    frame = run_sweep(arguments.config.resolve(), arguments.models, arguments.quick, arguments.output)
    print(frame.to_string(index=False))
