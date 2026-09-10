"""Recalculate and verify the ANN and GRU values reported in the final report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from parameter_sweep import evaluate_combination, load_test_windows, resolve_project_path


DEFAULT_CONFIG = Path(__file__).with_name("replication_config.yaml")


def verify(config_path: Path, tolerance: float) -> dict:
    cfg = OmegaConf.load(config_path)
    test_windows = load_test_windows(cfg)
    results_dir = resolve_project_path(cfg.paths.results_dir)
    verification = {"passed": True, "models": {}}

    for model in ("ANN", "GRU"):
        expected = cfg.reported_results[model]
        detail_path = results_dir / f"unweighted__{model}_anomaly_detection_results.csv"
        detailed = pd.read_csv(detail_path, parse_dates=["interval_start"]).set_index("interval_start")
        actual = evaluate_combination(
            detailed,
            test_windows,
            cfg.experiment.nab_profile,
            float(expected.threshold),
            int(expected.long_window),
            int(expected.short_window),
            use_reference_nab=True,
        )
        checks = {
            "true_positive_windows": actual["true_positive_windows"] == expected.true_positive_windows,
            "false_positive_alerts": actual["false_positive_alerts"] == expected.false_positive_alerts,
            "false_negative_windows": actual["false_negative_windows"] == expected.false_negative_windows,
            "normalized_nab_score": abs(
                actual["normalized_nab_score"] - expected.normalized_nab_score
            ) <= tolerance,
        }
        model_passed = all(checks.values())
        verification["passed"] = verification["passed"] and model_passed
        verification["models"][model] = {
            "passed": model_passed,
            "checks": checks,
            "actual": actual,
        }

    output_dir = resolve_project_path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "verification.json"
    output_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    return verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = verify(arguments.config.resolve(), arguments.tolerance)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
