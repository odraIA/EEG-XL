#!/usr/bin/env python3
"""Run staged EEG sweeps with checkpoint promotion between stages."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from make_eeg_sweep_plan import (
    command_for_candidate,
    dataset_label_for_task_mode,
    experiment_name,
    format_command,
    load_config,
    _checkpoint_for_tokenizer,
    _env_bool,
    _env_list,
    _parse_frequency_list,
)


def _plain_config(path: Path) -> dict[str, Any]:
    return load_config(path)


def _stage_dir_name(stage_index: int, stage_name: str) -> str:
    return f"stage_{stage_index}_{stage_name}"


def _best_is_better(candidate: float, incumbent: float | None, mode: str) -> bool:
    if incumbent is None:
        return True
    if mode == "min":
        return candidate < incumbent
    return candidate > incumbent


def _read_json_metric(path: Path, metric_name: str) -> float | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if metric_name in data and isinstance(data[metric_name], (int, float)):
        return float(data[metric_name])
    for key in ("val_metrics", "metrics", "test_metrics", "test_metrics_at_best_val"):
        nested = data.get(key)
        if isinstance(nested, dict) and isinstance(nested.get(metric_name), (int, float)):
            return float(nested[metric_name])
    return None


def _read_text_metric(path: Path, metric_name: str) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(metric_name):
            continue
        _, _, value = line.partition(":")
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _read_torch_checkpoint_metric(path: Path, metric_name: str) -> float | None:
    if not path.exists():
        return None
    try:
        import torch
    except Exception:
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return None
    for key in ("val_metrics", "metrics", "test_metrics_at_best_val", "best_test_metrics_at_best_val"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict) and isinstance(nested.get(metric_name), (int, float)):
            return float(nested[metric_name])
    value = checkpoint.get(metric_name)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def candidate_metric(candidate: dict[str, Any], metric_name: str) -> float:
    search_dirs = [Path(candidate["checkpoint_dir"]), Path(candidate["save_dir"])]
    json_names = ["metrics.json", "val_metrics.json", "final_results.json"]
    for directory in search_dirs:
        for name in json_names:
            metric = _read_json_metric(directory / name, metric_name)
            if metric is not None:
                return metric
        metric = _read_text_metric(directory / "final_results.txt", metric_name)
        if metric is not None:
            return metric
        metric = _read_torch_checkpoint_metric(directory / "checkpoint_best.pt", metric_name)
        if metric is not None:
            return metric
    raise FileNotFoundError(
        f"No metric {metric_name!r} found for {candidate['experiment_name']} "
        f"in {search_dirs}"
    )


def candidate_checkpoint(candidate: dict[str, Any]) -> Path:
    candidates = [
        Path(candidate["checkpoint_dir"]) / "checkpoint_best.pt",
        Path(candidate["save_dir"]) / "checkpoint_best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"checkpoint_best.pt not found for {candidate['experiment_name']} "
        f"in {[str(path.parent) for path in candidates]}"
    )


def promote_best_candidate(
    *,
    stage_index: int,
    stage_name: str,
    candidates: list[dict[str, Any]],
    metric_name: str,
    metric_mode: str,
    promotion_root: Path,
    promotion_record_dir: Path,
    previous_promoted_checkpoint: str | None = None,
    lineage_so_far: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    best_candidate: dict[str, Any] | None = None
    best_metric: float | None = None
    scored: list[dict[str, Any]] = []

    for candidate in candidates:
        if candidate.get("status") == "failed":
            scored.append(
                {
                    "experiment_name": candidate["experiment_name"],
                    "status": "failed",
                    "return_code": candidate.get("return_code"),
                    "checkpoint_dir": candidate["checkpoint_dir"],
                }
            )
            continue
        metric = candidate_metric(candidate, metric_name)
        scored.append(
            {
                "experiment_name": candidate["experiment_name"],
                "status": candidate.get("status", "completed"),
                "metric": metric,
                "checkpoint_dir": candidate["checkpoint_dir"],
                "final_results_path": str(Path(candidate["save_dir"]) / "final_results.json"),
                "config_snapshot": str(Path(candidate["save_dir"]) / "config_resolved.yaml"),
            }
        )
        if _best_is_better(metric, best_metric, metric_mode):
            best_metric = metric
            best_candidate = candidate

    if best_candidate is None or best_metric is None:
        raise RuntimeError(f"No promotable candidate found for stage {stage_name}")

    source = candidate_checkpoint(best_candidate)
    stage_dir = promotion_root / _stage_dir_name(stage_index, stage_name)
    destination = stage_dir / "best_checkpoint.pt"
    record_path = promotion_record_dir / f"{_stage_dir_name(stage_index, stage_name)}_promotion.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite promoted checkpoint: {destination}")
    if record_path.exists():
        raise FileExistsError(f"Refusing to overwrite promotion record: {record_path}")

    stage_dir.mkdir(parents=True, exist_ok=True)
    promotion_record_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    record = {
        "stage_index": stage_index,
        "stage_name": stage_name,
        "metric_name": metric_name,
        "metric_mode": metric_mode,
        "selected_metric": best_metric,
        "selected_metric_value": best_metric,
        "selected_candidate": best_candidate,
        "selected_experiment": best_candidate["experiment_name"],
        "scored_candidates": scored,
        "completed_candidates": [
            item for item in scored
            if item.get("status") not in {"failed"}
        ],
        "failed_candidates": [
            item for item in scored
            if item.get("status") == "failed"
        ],
        "source_checkpoint": str(source),
        "source_experiment": best_candidate["experiment_name"],
        "promoted_checkpoint": str(destination),
        "previous_promoted_checkpoint": previous_promoted_checkpoint,
        "config_snapshot": str(Path(best_candidate["save_dir"]) / "config_resolved.yaml"),
        "lineage": lineage_so_far or [],
    }
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    record["promotion_record"] = str(record_path)
    return record


def _candidate_values(state: dict[str, Any], stage: dict[str, Any]) -> list[dict[str, Any]]:
    variations: list[dict[str, Any]] = []
    if "target_sfreq" in stage:
        variations.extend({"target_sfreq": value} for value in stage["target_sfreq"])
    if "tokenizers" in stage:
        variations.extend({"tokenizer": value} for value in stage["tokenizers"])
    if "task_modes" in stage:
        variations.extend({"task_mode": value} for value in stage["task_modes"])
    if "seeds" in stage:
        variations.extend({"seed": value} for value in stage["seeds"])
    if not variations:
        variations.append({})

    candidates = []
    for variation in variations:
        candidate = dict(state)
        candidate.update(variation)
        candidates.append(candidate)
    return candidates


def build_stage_candidates(
    cfg: dict[str, Any],
    stage_index: int,
    stage: dict[str, Any],
    state: dict[str, Any],
    previous_promoted: str | None,
    output_root: Path,
    checkpoint_root: Path,
) -> list[dict[str, Any]]:
    candidates = []
    for candidate_state in _candidate_values(state, stage):
        task_mode = candidate_state["task_mode"]
        tokenizer = candidate_state["tokenizer"]
        initialization = "promoted" if previous_promoted else candidate_state["initialization"]
        experiment = experiment_name(
            task_mode,
            candidate_state["target_sfreq"],
            tokenizer["name"],
            initialization,
            int(candidate_state["seed"]),
            prefix=f"eeg_chain_s{stage_index}_{stage['name']}",
        )
        save_dir = str(output_root / experiment)
        checkpoint_dir = str(checkpoint_root / experiment)
        command = command_for_candidate(
            python_module=cfg.get("python_module", "brainstorm.evaluate_criss_cross_word_classification"),
            base_config=cfg["base_configs"][task_mode],
            task_mode=task_mode,
            target_sfreq=candidate_state["target_sfreq"],
            tokenizer_name=tokenizer["name"],
            tokenizer_checkpoint=_checkpoint_for_tokenizer(tokenizer),
            initialization=candidate_state["initialization"],
            seed=int(candidate_state["seed"]),
            experiment=experiment,
            save_dir=save_dir,
            checkpoint_dir=checkpoint_dir,
            promoted_checkpoint=previous_promoted,
            criss_cross_checkpoint=os.environ.get("CRISS_CROSS_CHECKPOINT") or None,
        )
        candidates.append(
            {
                "stage_index": stage_index,
                "stage_name": stage["name"],
                "experiment_name": experiment,
                "dataset_label": dataset_label_for_task_mode(task_mode),
                "task_mode": task_mode,
                "target_sfreq": candidate_state["target_sfreq"],
                "tokenizer_name": tokenizer["name"],
                "tokenizer_checkpoint": _checkpoint_for_tokenizer(tokenizer),
                "initialization": initialization,
                "seed": int(candidate_state["seed"]),
                "base_config": cfg["base_configs"][task_mode],
                "save_dir": save_dir,
                "checkpoint_dir": checkpoint_dir,
                "previous_promoted_checkpoint": previous_promoted,
                "command": command,
                "_state": candidate_state,
            }
        )
    return candidates


def _write_simulated_outputs(candidates: list[dict[str, Any]], metric_name: str, stage_index: int) -> None:
    for candidate_idx, candidate in enumerate(candidates):
        checkpoint_dir = Path(candidate["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metric = float(stage_index * 100 + candidate_idx)
        (checkpoint_dir / "metrics.json").write_text(
            json.dumps({metric_name: metric}, indent=2),
            encoding="utf-8",
        )
        (checkpoint_dir / "checkpoint_best.pt").write_text(
            f"fake checkpoint for {candidate['experiment_name']}\n",
            encoding="utf-8",
        )


def run_or_print_candidates(
    candidates: list[dict[str, Any]],
    *,
    dry_run: bool,
    continue_on_error: bool,
) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        print(f"# [{idx}/{len(candidates)}] {candidate['experiment_name']}")
        print(format_command(candidate["command"]))
        if dry_run:
            candidate["status"] = "dry_run"
            candidate["return_code"] = None
            statuses.append(candidate)
            continue
        completed = subprocess.run(candidate["command"], check=False)
        candidate["return_code"] = completed.returncode
        if completed.returncode != 0:
            candidate["status"] = "failed"
            message = (
                f"Candidate {candidate['experiment_name']} failed with "
                f"exit code {completed.returncode}"
            )
            if continue_on_error:
                print(f"# {message}")
                statuses.append(candidate)
                continue
            raise SystemExit(message)
        candidate["status"] = "completed"
        statuses.append(candidate)
    return statuses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eeg_chained_sweep.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Print staged commands without running or promoting.")
    parser.add_argument("--limit", type=int, default=None, help="Limit printed candidates per stage in dry-run mode.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--simulate-promotions", action="store_true", help="Create fake metrics/checkpoints and test promotion logic.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Root for simulated candidate outputs.")
    parser.add_argument("--promotion-root", type=Path, default=None)
    parser.add_argument("--promotion-record-dir", type=Path, default=None)
    parser.add_argument("--lineage-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _plain_config(args.config)
    metric_name = os.environ.get("EEG_SELECTION_METRIC") or cfg.get("metric_name", "balanced_top10_accuracy_retrieval250")
    metric_mode = os.environ.get("EEG_SELECTION_MODE") or cfg.get("metric_mode", "max")

    env_frequencies = _env_list("EEG_SWEEP_FREQUENCIES")
    if env_frequencies:
        for stage in cfg.get("stages", []):
            if "target_sfreq" in stage:
                stage["target_sfreq"] = _parse_frequency_list(env_frequencies)

    if args.simulate_promotions:
        work_dir = args.work_dir or Path(f"/tmp/scrabrain_eeg_chain_smoke_{int(time.time())}")
        output_root = work_dir / "logs"
        checkpoint_root = work_dir / "checkpoints"
    else:
        output_root = Path(cfg.get("output_root", "./logs/eeg_chained_sweeps"))
        checkpoint_root = Path(cfg.get("checkpoint_root", "./checkpoints/eeg_chained_sweeps"))

    promotion_root = args.promotion_root or Path(cfg.get("promotion_root", "./checkpoints/eeg_promoted"))
    promotion_record_dir = args.promotion_record_dir or Path(cfg.get("promotion_record_dir", "./promotions"))
    lineage_dir = args.lineage_dir or Path(cfg.get("lineage_dir", "./promotions"))

    defaults = cfg["defaults"]
    if os.environ.get("EEG_TARGET_SFREQ"):
        raw_sfreq = float(os.environ["EEG_TARGET_SFREQ"])
        defaults["target_sfreq"] = int(raw_sfreq) if raw_sfreq.is_integer() else raw_sfreq
    if os.environ.get("EEG_TOKENIZER_NAME"):
        defaults["tokenizer"]["name"] = os.environ["EEG_TOKENIZER_NAME"]
    if os.environ.get("EEG_TRAIN_FROM_SCRATCH"):
        defaults["initialization"] = "scratch" if _env_bool("EEG_TRAIN_FROM_SCRATCH") else "pretrained"
    if os.environ.get("EEG_PROMOTED_CHECKPOINT"):
        defaults["initialization"] = "promoted"
    defaults["tokenizer"]["checkpoint"] = _checkpoint_for_tokenizer(defaults["tokenizer"])

    state = {
        "target_sfreq": defaults["target_sfreq"],
        "task_mode": defaults["task_mode"],
        "tokenizer": defaults["tokenizer"],
        "initialization": defaults.get("initialization", "pretrained"),
        "seed": defaults.get("seed", 42),
    }
    previous_promoted: str | None = None
    lineage: list[dict[str, Any]] = []

    for stage_index, stage in enumerate(cfg["stages"], start=1):
        candidates = build_stage_candidates(
            cfg,
            stage_index,
            stage,
            state,
            previous_promoted,
            output_root,
            checkpoint_root,
        )
        visible = candidates[: args.limit] if args.limit and args.dry_run else candidates
        print(f"# Stage {stage_index}: {stage['name']} ({len(visible)} of {len(candidates)} candidates)")

        if args.simulate_promotions:
            _write_simulated_outputs(candidates, metric_name, stage_index)
        else:
            run_or_print_candidates(
                visible,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
            )

        if args.dry_run and not args.simulate_promotions:
            continue

        promotion = promote_best_candidate(
            stage_index=stage_index,
            stage_name=stage["name"],
            candidates=candidates,
            metric_name=metric_name,
            metric_mode=metric_mode,
            promotion_root=promotion_root,
            promotion_record_dir=promotion_record_dir,
            previous_promoted_checkpoint=previous_promoted,
            lineage_so_far=lineage,
        )
        lineage.append(promotion)
        selected_state = dict(promotion["selected_candidate"]["_state"])
        state.update(selected_state)
        previous_promoted = promotion["promoted_checkpoint"]
        print(
            f"# Promoted {promotion['selected_candidate']['experiment_name']} "
            f"({metric_name}={promotion['selected_metric']}) -> {previous_promoted}"
        )

    if not args.dry_run or args.simulate_promotions:
        lineage_dir.mkdir(parents=True, exist_ok=True)
        lineage_path = lineage_dir / "eeg_chained_sweep_lineage.json"
        if lineage_path.exists():
            raise FileExistsError(f"Refusing to overwrite lineage file: {lineage_path}")
        lineage_path.write_text(
            json.dumps(
                {
                    "config": str(args.config),
                    "metric_name": metric_name,
                    "metric_mode": metric_mode,
                    "selected_stages": lineage,
                    "final_promoted_checkpoint": previous_promoted,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"# Lineage saved to {lineage_path}")


if __name__ == "__main__":
    main()
