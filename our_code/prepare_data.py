"""Prepare and document the IBM Cloud features used in our replication."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("replication_config.yaml")


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic hour and weekday features without changing the input frame."""
    result = frame.copy()
    result["sin_hour"] = np.sin(2 * np.pi * result.index.hour / 24)
    result["cos_hour"] = np.cos(2 * np.pi * result.index.hour / 24)
    result["sin_day_of_week"] = np.sin(2 * np.pi * result.index.dayofweek / 7)
    result["cos_day_of_week"] = np.cos(2 * np.pi * result.index.dayofweek / 7)
    return result


def prepare(config_path: Path, output_override: Path | None = None) -> dict:
    cfg = OmegaConf.load(config_path)
    raw_path = resolve_project_path(cfg.paths.raw_data)
    labels_path = resolve_project_path(cfg.paths.labels)
    output_path = output_override or resolve_project_path(cfg.paths.prepared_data)
    timestamp = cfg.preprocessing.timestamp_column
    pattern = re.compile(cfg.preprocessing.feature_pattern)

    parquet_file = pq.ParquetFile(raw_path)
    raw_columns = parquet_file.schema.names
    feature_columns = [name for name in raw_columns if pattern.search(name)]
    selected_columns = [*feature_columns]
    if timestamp in raw_columns:
        selected_columns.append(timestamp)
    else:
        raise ValueError(f"Timestamp column '{timestamp}' is missing from {raw_path}")

    frame = pd.read_parquet(raw_path, columns=selected_columns)
    missing_before = int(frame[feature_columns].isna().sum().sum())
    frame[feature_columns] = frame[feature_columns].fillna(0)
    frame[timestamp] = pd.to_datetime(frame[timestamp], unit="s")
    frame = frame.set_index(timestamp).sort_index()
    frame = add_time_features(frame)

    start = pd.Timestamp(cfg.experiment.start_date)
    train_end = pd.Timestamp(cfg.experiment.train_end_date)
    test_start = pd.Timestamp(cfg.experiment.test_start_date)
    end_exclusive = pd.Timestamp(cfg.experiment.end_date) + pd.Timedelta(days=1)
    experiment_frame = frame.loc[start:end_exclusive]
    train_rows = len(experiment_frame.loc[start:train_end])
    test_rows = len(experiment_frame.loc[test_start:end_exclusive])

    labels = pd.read_csv(labels_path)
    labels["anomaly_start"] = pd.to_datetime(labels["anomaly_start"], utc=True).dt.tz_localize(None)
    labels["anomaly_end"] = pd.to_datetime(labels["anomaly_end"], utc=True).dt.tz_localize(None)
    labels["window_start"] = labels["anomaly_start"] - pd.Timedelta(
        minutes=int(cfg.experiment.anomaly_minutes_before)
    )
    test_windows = labels.loc[
        (labels["window_start"] >= test_start) & (labels["anomaly_end"] <= end_exclusive)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path)
    metadata = {
        "raw_data": str(raw_path),
        "prepared_data": str(output_path),
        "raw_column_count": len(raw_columns),
        "selected_5xx_feature_count": len(feature_columns),
        "added_time_feature_count": 4,
        "final_feature_count": len(frame.columns),
        "row_count": len(frame),
        "missing_values_replaced": missing_before,
        "train_row_count": train_rows,
        "test_row_count": test_rows,
        "test_anomaly_window_count": len(test_windows),
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="Optional prepared Parquet output path")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    summary = prepare(arguments.config.resolve(), arguments.output)
    print(json.dumps(summary, indent=2))
