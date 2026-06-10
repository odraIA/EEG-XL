#!/usr/bin/env python3
"""Summarize EEG experiment and chained sweep artifacts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = BASE_DIR / "logs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
PROMOTIONS_DIR = BASE_DIR / "promotions"

EXPERIMENT_COLUMNS = [
    "experiment",
    "dataset",
    "task_mode",
    "target_sfreq",
    "tokenizer",
    "training_mode",
    "promoted_checkpoint",
    "seed",
    "best_val_epoch",
    "best_val_metric",
    "test_metric",
    "checkpoint_path",
    "final_results_path",
]

CHAINED_COLUMNS = [
    "stage",
    "stage_name",
    "selected_experiment",
    "selected_metric",
    "selected_metric_value",
    "source_checkpoint",
    "promoted_checkpoint",
    "previous_promoted_checkpoint",
    "completed_candidates",
    "failed_candidates",
    "config_snapshot",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _parse_final_results_txt(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"\s*([A-Za-z0-9_./-]+)\s*:\s*([-+0-9.eE]+)\s*$", line)
        if not match:
            continue
        try:
            metrics[match.group(1)] = float(match.group(2))
        except ValueError:
            pass
    return metrics


def _latest_epoch_row(run_dir: Path) -> dict[str, str]:
    for name in ("epoch_metrics.csv", "metrics_history.csv"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            return rows[-1] if rows else {}
        except Exception:
            return {}
    return {}


def _pick_metric(metrics: dict[str, Any]) -> Any:
    if not metrics:
        return ""
    preferred = [
        "balanced_top10_accuracy_retrieval250",
        "top10_accuracy_retrieval250",
        "balanced_top10_accuracy_retrieval50",
        "top10_accuracy_retrieval50",
    ]
    for key in preferred:
        if key in metrics:
            return metrics[key]
    for key, value in metrics.items():
        if "loss" not in key.lower() and isinstance(value, (int, float)):
            return value
    return ""


def _dataset_label(metadata: dict[str, Any]) -> str:
    datasets = metadata.get("datasets")
    if isinstance(datasets, list) and datasets:
        return "+".join(
            str(item.get("dataset_name") or item.get("dataset_type") or "")
            for item in datasets
            if isinstance(item, dict)
        )
    return str(metadata.get("dataset_type") or "")


def _discover_run_dirs() -> set[Path]:
    dirs: set[Path] = set()
    for root in (LOGS_DIR, RESULTS_DIR):
        if root.exists():
            for marker in ("run_metadata.json", "final_results.json", "final_results.txt", "epoch_metrics.csv"):
                dirs.update(path.parent for path in root.rglob(marker))
    if CHECKPOINTS_DIR.exists():
        for path in CHECKPOINTS_DIR.rglob("checkpoint_best.pt"):
            dirs.add(path.parent)
    return dirs


def experiment_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run_dir in sorted(_discover_run_dirs()):
        metadata = _read_json(run_dir / "run_metadata.json")
        final_json = _read_json(run_dir / "final_results.json")
        final_txt = run_dir / "final_results.txt"
        txt_metrics = _parse_final_results_txt(final_txt)
        final_metrics = final_json.get("test_metrics") if isinstance(final_json.get("test_metrics"), dict) else {}
        metrics = {**txt_metrics, **final_metrics}
        latest_epoch = _latest_epoch_row(run_dir)

        experiment = (
            metadata.get("experiment_name")
            or final_json.get("experiment_name")
            or run_dir.name
        )
        if str(experiment) in seen:
            continue
        seen.add(str(experiment))

        checkpoint_best = metadata.get("checkpoint_paths", {}).get("checkpoint_best", "") if isinstance(metadata.get("checkpoint_paths"), dict) else ""
        if not checkpoint_best:
            checkpoint_best = str(run_dir / "checkpoint_best.pt") if (run_dir / "checkpoint_best.pt").exists() else ""

        rows.append({
            "experiment": experiment,
            "dataset": _dataset_label(metadata),
            "task_mode": metadata.get("task_mode") or final_json.get("task_mode") or "",
            "target_sfreq": metadata.get("target_sfreq") or final_json.get("target_sfreq") or "",
            "tokenizer": (metadata.get("tokenizer") or {}).get("name", "") if isinstance(metadata.get("tokenizer"), dict) else final_json.get("tokenizer_name", ""),
            "training_mode": metadata.get("training_mode") or final_json.get("training_mode") or "",
            "promoted_checkpoint": (metadata.get("checkpoint_paths") or {}).get("promoted_checkpoint", "") if isinstance(metadata.get("checkpoint_paths"), dict) else "",
            "seed": metadata.get("seed") or final_json.get("seed") or "",
            "best_val_epoch": latest_epoch.get("val/best_epoch") or latest_epoch.get("epoch") or "",
            "best_val_metric": latest_epoch.get("val/best_primary_metric") or latest_epoch.get("val/primary_metric_value") or "",
            "test_metric": _pick_metric(metrics),
            "checkpoint_path": checkpoint_best,
            "final_results_path": str(run_dir / "final_results.json" if (run_dir / "final_results.json").exists() else final_txt),
        })
    return rows


def chained_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not PROMOTIONS_DIR.exists():
        return rows
    for path in sorted(PROMOTIONS_DIR.glob("stage_*_promotion.json")):
        data = _read_json(path)
        rows.append({
            "stage": data.get("stage_index", ""),
            "stage_name": data.get("stage_name", path.stem),
            "selected_experiment": data.get("selected_experiment") or (data.get("selected_candidate") or {}).get("experiment_name", ""),
            "selected_metric": data.get("metric_name", ""),
            "selected_metric_value": data.get("selected_metric_value", data.get("selected_metric", "")),
            "source_checkpoint": data.get("source_checkpoint", ""),
            "promoted_checkpoint": data.get("promoted_checkpoint", ""),
            "previous_promoted_checkpoint": data.get("previous_promoted_checkpoint", ""),
            "completed_candidates": len(data.get("completed_candidates", []) or []),
            "failed_candidates": len(data.get("failed_candidates", []) or []),
            "config_snapshot": data.get("config_snapshot", ""),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(col, "")).replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    exp_rows = experiment_rows()
    chain_rows = chained_rows()

    write_csv(RESULTS_DIR / "eeg_experiments_summary.csv", exp_rows, EXPERIMENT_COLUMNS)
    write_markdown(RESULTS_DIR / "eeg_experiments_summary.md", exp_rows, EXPERIMENT_COLUMNS)
    write_csv(RESULTS_DIR / "chained_eeg_sweep_summary.csv", chain_rows, CHAINED_COLUMNS)
    write_markdown(RESULTS_DIR / "chained_eeg_sweep_summary.md", chain_rows, CHAINED_COLUMNS)

    print(f"Wrote {len(exp_rows)} EEG experiment rows")
    print(f"Wrote {len(chain_rows)} chained sweep rows")


if __name__ == "__main__":
    main()
