#!/usr/bin/env python3
"""Run or dry-run independent EEG CrissCross sweeps."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from make_eeg_sweep_plan import build_plan, format_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eeg_sweep.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching training.")
    parser.add_argument("--limit", type=int, default=None, help="Run or print only the first N candidates.")
    parser.add_argument("--task-mode", default=None)
    parser.add_argument("--dataset", default=None, help="Dataset group name from configs/eeg_sweep.yaml.")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--initialization", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(
        args.config,
        task_mode_filter=args.task_mode,
        dataset_filter=args.dataset,
        tokenizer_filter=args.tokenizer,
        initialization_filter=args.initialization,
    )
    selected = plan[: args.limit] if args.limit is not None else plan

    print(f"# {len(selected)} of {len(plan)} EEG sweep candidates")
    for idx, candidate in enumerate(selected, start=1):
        command = candidate["command"]
        print(f"# [{idx}/{len(selected)}] {candidate['experiment_name']}")
        print(format_command(command))
        if args.dry_run:
            continue
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            message = (
                f"Candidate {candidate['experiment_name']} failed with "
                f"exit code {completed.returncode}"
            )
            if args.continue_on_error:
                print(f"# {message}")
                continue
            raise SystemExit(message)


if __name__ == "__main__":
    main()
