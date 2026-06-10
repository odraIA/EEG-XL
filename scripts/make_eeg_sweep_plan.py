#!/usr/bin/env python3
"""Build independent EEG CrissCross sweep commands."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shlex
from itertools import product
from pathlib import Path
from typing import Any

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - fallback for lightweight dry-run envs
    OmegaConf = None


TASK_DATASET_LABELS = {
    "reading": "zuco-eegdash",
    "listening": "ds004408-ds007808",
    "reading_listening": "zuco-eegdash-ds004408-ds007808",
}

TOKENIZER_CHECKPOINT_ENVS = {
    "biocodec": "BIOCODEC_CHECKPOINT",
    "brainomni_base": "BRAINOMNI_CHECKPOINT",
    "brainomni_tiny": "BRAINOMNI_CHECKPOINT",
    "braintokenizer": "BRAINOMNI_CHECKPOINT",
}


def _env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> list[str] | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_frequency_list(values: list[str]) -> list[int | float]:
    parsed: list[int | float] = []
    for value in values:
        numeric = float(value)
        parsed.append(int(numeric) if numeric.is_integer() else numeric)
    return parsed


def _checkpoint_for_tokenizer(tokenizer: dict[str, Any]) -> str | None:
    env_name = TOKENIZER_CHECKPOINT_ENVS.get(str(tokenizer.get("name", "")).lower())
    if env_name and os.environ.get(env_name):
        return os.environ[env_name]
    return tokenizer.get("checkpoint")


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx].rstrip()
    return line.rstrip()


def _split_key_value(text: str) -> tuple[str, str | None]:
    key, sep, value = text.partition(":")
    if not sep:
        raise ValueError(f"Invalid YAML mapping line: {text}")
    value = value.strip()
    return key.strip(), value if value else None


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "None", "~"}:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except Exception:
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _yaml_lines(path: Path) -> list[tuple[int, str]]:
    parsed = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = _strip_comment(raw_line)
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        parsed.append((indent, stripped.strip()))
    return parsed


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    is_list = lines[index][1].startswith("- ")
    if is_list:
        items = []
        while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
            rest = lines[index][1][2:].strip()
            index += 1
            if not rest:
                item, index = _parse_block(lines, index, lines[index][0])
                items.append(item)
                continue
            if ":" in rest:
                item = {}
                key, value = _split_key_value(rest)
                if value is None:
                    child, index = _parse_block(lines, index, lines[index][0])
                    item[key] = child
                else:
                    item[key] = _parse_scalar(value)
                while index < len(lines) and lines[index][0] > indent:
                    extra, index = _parse_block(lines, index, lines[index][0])
                    if not isinstance(extra, dict):
                        raise ValueError(f"Expected mapping inside list item near {lines[index - 1]}")
                    item.update(extra)
                items.append(item)
            else:
                items.append(_parse_scalar(rest))
        return items, index

    result = {}
    while index < len(lines) and lines[index][0] == indent and not lines[index][1].startswith("- "):
        key, value = _split_key_value(lines[index][1])
        index += 1
        if value is None:
            if index >= len(lines) or lines[index][0] <= indent:
                result[key] = {}
            else:
                child, index = _parse_block(lines, index, lines[index][0])
                result[key] = child
        else:
            result[key] = _parse_scalar(value)
    return result, index


def load_config(path: Path) -> dict[str, Any]:
    if OmegaConf is not None:
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    lines = _yaml_lines(path)
    parsed, index = _parse_block(lines, 0, lines[0][0] if lines else 0)
    if index != len(lines):
        raise ValueError(f"Could not parse full YAML file: {path}")
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected top-level mapping in {path}")
    return parsed


def _sfreq_label(value: Any) -> str:
    numeric = float(value)
    return f"{int(numeric)}hz" if numeric.is_integer() else f"{numeric:g}hz"


def _bool_override(value: bool) -> str:
    return "true" if value else "false"


def dataset_label_for_task_mode(task_mode: str) -> str:
    return TASK_DATASET_LABELS.get(task_mode, task_mode.replace("_", "-"))


def experiment_name(
    task_mode: str,
    target_sfreq: Any,
    tokenizer_name: str,
    initialization: str,
    seed: int,
    prefix: str = "eeg",
    dataset_label: str | None = None,
) -> str:
    return "_".join(
        [
            prefix,
            dataset_label or dataset_label_for_task_mode(task_mode),
            task_mode,
            _sfreq_label(target_sfreq),
            tokenizer_name,
            initialization,
            f"seed{seed}",
        ]
    )


def command_for_candidate(
    *,
    python_module: str,
    base_config: str,
    task_mode: str,
    target_sfreq: Any,
    tokenizer_name: str,
    tokenizer_checkpoint: str | None,
    initialization: str,
    seed: int,
    experiment: str,
    save_dir: str,
    checkpoint_dir: str,
    promoted_checkpoint: str | None = None,
    criss_cross_checkpoint: str | None = None,
) -> list[str]:
    train_from_scratch = initialization == "scratch" and promoted_checkpoint is None
    use_promoted = promoted_checkpoint is not None
    command = [
        "python",
        "-m",
        python_module,
        "--config-name",
        base_config,
        f"data.task_mode={task_mode}",
        f"data.target_sfreq={target_sfreq}",
        f"model.tokenizer_name={tokenizer_name}",
        f"model.train_from_scratch={_bool_override(train_from_scratch)}",
        f"model.use_promoted_checkpoint={_bool_override(use_promoted)}",
        f"seed={seed}",
        f"logging.experiment_name={experiment}",
        f"logging.save_dir={save_dir}",
        f"logging.checkpoint_dir={checkpoint_dir}",
    ]
    if criss_cross_checkpoint:
        command.append(f"model.criss_cross_checkpoint={criss_cross_checkpoint}")
    if tokenizer_checkpoint:
        command.append(f"model.tokenizer_checkpoint={tokenizer_checkpoint}")
    if promoted_checkpoint:
        command.append(f"model.promoted_checkpoint={promoted_checkpoint}")
    return command


def build_plan(
    config_path: Path,
    *,
    task_mode_filter: str | None = None,
    dataset_filter: str | None = None,
    tokenizer_filter: str | None = None,
    initialization_filter: str | None = None,
) -> list[dict[str, Any]]:
    plain = load_config(config_path)
    env_frequencies = _env_list("EEG_SWEEP_FREQUENCIES")
    if env_frequencies:
        plain["target_sfreq"] = _parse_frequency_list(env_frequencies)

    if task_mode_filter is None and os.environ.get("EEG_TASK_MODE"):
        task_mode_filter = os.environ["EEG_TASK_MODE"]
    if tokenizer_filter is None and os.environ.get("EEG_TOKENIZER_NAME"):
        tokenizer_filter = os.environ["EEG_TOKENIZER_NAME"]

    env_train_from_scratch = _env_bool("EEG_TRAIN_FROM_SCRATCH")
    env_promoted_checkpoint = os.environ.get("EEG_PROMOTED_CHECKPOINT") or None
    env_use_promoted = _env_bool("EEG_USE_PROMOTED_CHECKPOINT")
    if env_use_promoted and not env_promoted_checkpoint:
        env_promoted_checkpoint = os.environ.get("EEG_PROMOTED_CHECKPOINT") or os.environ.get("CRISS_CROSS_CHECKPOINT")
    criss_cross_checkpoint = os.environ.get("CRISS_CROSS_CHECKPOINT") or None

    base_configs = plain["base_configs"]
    dataset_groups = plain.get("dataset_groups")
    if dataset_groups:
        groups = list(dataset_groups)
    else:
        groups = [
            {
                "name": dataset_label_for_task_mode(task_mode),
                "task_mode": task_mode,
                "base_config": base_configs[task_mode],
            }
            for task_mode in plain.get("task_modes", base_configs.keys())
        ]

    if task_mode_filter:
        groups = [group for group in groups if group["task_mode"] == task_mode_filter]
    if dataset_filter:
        groups = [group for group in groups if group["name"] == dataset_filter]

    tokenizers = list(plain["tokenizers"])
    if tokenizer_filter:
        tokenizers = [tok for tok in tokenizers if tok["name"] == tokenizer_filter]

    initializations = list(plain.get("initializations", ["pretrained"]))
    if env_promoted_checkpoint:
        initializations = ["promoted"]
    elif env_train_from_scratch is not None:
        initializations = ["scratch" if env_train_from_scratch else "pretrained"]
    if initialization_filter:
        initializations = [init for init in initializations if init == initialization_filter]

    output_root = Path(plain.get("output_root", "./logs/eeg_sweeps"))
    checkpoint_root = Path(plain.get("checkpoint_root", "./checkpoints/eeg_sweeps"))
    python_module = plain.get("python_module", "brainstorm.evaluate_criss_cross_word_classification")

    plan: list[dict[str, Any]] = []
    for dataset_group, target_sfreq, tokenizer, initialization, seed in product(
        groups,
        plain["target_sfreq"],
        tokenizers,
        initializations,
        plain.get("seeds", [42]),
    ):
        task_mode = dataset_group["task_mode"]
        base_config = dataset_group.get("base_config", base_configs.get(task_mode))
        if not base_config:
            raise KeyError(f"No base config registered for dataset group={dataset_group!r}")
        experiment = experiment_name(
            task_mode,
            target_sfreq,
            tokenizer["name"],
            initialization,
            int(seed),
            dataset_label=dataset_group["name"],
        )
        save_dir = str(output_root / experiment)
        checkpoint_dir = str(checkpoint_root / experiment)
        tokenizer_checkpoint = _checkpoint_for_tokenizer(tokenizer)
        command = command_for_candidate(
            python_module=python_module,
            base_config=base_config,
            task_mode=task_mode,
            target_sfreq=target_sfreq,
            tokenizer_name=tokenizer["name"],
            tokenizer_checkpoint=tokenizer_checkpoint,
            initialization=initialization,
            seed=int(seed),
            experiment=experiment,
            save_dir=save_dir,
            checkpoint_dir=checkpoint_dir,
            promoted_checkpoint=env_promoted_checkpoint,
            criss_cross_checkpoint=criss_cross_checkpoint,
        )
        plan.append(
            {
                "experiment_name": experiment,
                "dataset_group": dataset_group["name"],
                "dataset_label": dataset_group["name"],
                "task_mode": task_mode,
                "target_sfreq": target_sfreq,
                "tokenizer_name": tokenizer["name"],
                "tokenizer_checkpoint": tokenizer_checkpoint,
                "initialization": initialization,
                "seed": int(seed),
                "base_config": base_config,
                "save_dir": save_dir,
                "checkpoint_dir": checkpoint_dir,
                "promoted_checkpoint": env_promoted_checkpoint,
                "criss_cross_checkpoint": criss_cross_checkpoint,
                "command": command,
            }
        )
    return plan


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eeg_sweep.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running anything.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON plan output path.")
    parser.add_argument("--format", choices=("shell", "json", "jsonl"), default="shell")
    parser.add_argument("--limit", type=int, default=None, help="Limit printed/output candidates.")
    parser.add_argument("--task-mode", default=None)
    parser.add_argument("--dataset", default=None, help="Dataset group name from configs/eeg_sweep.yaml.")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--initialization", default=None)
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

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    if args.format == "json":
        print(json.dumps(selected, indent=2))
    elif args.format == "jsonl":
        for item in selected:
            print(json.dumps(item, sort_keys=True))
    else:
        print(f"# {len(selected)} of {len(plan)} EEG sweep candidates")
        for item in selected:
            print(f"# {item['experiment_name']}")
            print(format_command(item["command"]))

    if not args.dry_run and args.output is None:
        print("# Plan generation only. Use scripts/run_eeg_sweep.py to execute candidates.")


if __name__ == "__main__":
    main()
