#!/usr/bin/env python3
"""Read-only web monitor for scripts/run_ds004408_four_way_finetuning.sh.

Everything is derived from files the pipeline already writes, so the monitor
needs no Docker socket and can safely be exposed as a public page:

  results/ds004408_four_way/<RUN_ID>/runs.tsv, current_containers.tsv, COMPLETED
  logs/ds004408_four_way_<RUN_ID>.log                       (master log)
  logs/word_classification_ds004408_four_way/<RUN_ID>/<label>/seed<N>/
      epoch_metrics.jsonl, stdout_stderr.log, final_results.json

Standard library only. Run with:
  python scripts/monitor_ds004408_four_way.py --root . --port 8093
or through docker-compose.ds004408-monitor.yml.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

JOB_LABELS = ["random_init", "eeg_from_scratch", "megxl_eeg2", "megxl_eeg1"]
JOB_EMBEDDING_IDS = {"random_init": 2, "eeg_from_scratch": 2, "megxl_eeg2": 2, "megxl_eeg1": 1}
MODEL_NAMES = {
    "random_init": "Random init",
    "eeg_from_scratch": "EEG curriculum (scratch)",
    "megxl_eeg2": "MEG-XL → EEG (EEG row 2)",
    "megxl_eeg1": "MEG-XL → EEG (MEG row 1)",
}
PRIMARY_TEST_METRICS = [
    "balanced_top10_accuracy_retrieval250",
    "balanced_top10_accuracy_retrieval50",
]

SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
ASCTIME = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - ")
EPOCH_LINE = re.compile(r"^\s*Epoch (\d+)/(\d+)\s*$")
TQDM = re.compile(
    r"(Training|Evaluating[^:]*|[A-Za-z][A-Za-z _-]{0,40}):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*"
    r"\[([0-9:]+)<([0-9:?]+)"
)
SECRETS = re.compile(
    r"(hf_[A-Za-z0-9]{12,}|(?i:(?:api[_-]?key|token|password|secret)\s*[=:]\s*)\S+)"
)


class Config:
    root = Path("/workspace")
    results_base = "results/ds004408_four_way"
    logs_base = "logs/word_classification_ds004408_four_way"
    master_log_pattern = "logs/ds004408_four_way_{run_id}.log"
    stale_minutes = 20.0
    default_epochs = 50
    default_patience = 10
    show_logs = True
    title = "ds004408 four-way fine-tuning"
    gpu_dashboard_url = ""


CFG = Config()


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #

def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_tsv(path: Path) -> list[dict]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except OSError:
        return []


def _read_csv(path: Path) -> list[dict]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _read_tail(path: Path, max_bytes: int) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _num(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _iso_to_ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _asctime_to_ts(value: str) -> float:
    # Only differences between these timestamps are used, so the container
    # timezone does not matter.
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").timestamp()


class IncrementalScanner:
    """Feeds only newly appended lines of a growing log file to a parser."""

    def __init__(self, path: Path, factory):
        self.path = path
        self.factory = factory
        self.offset = 0
        self.inode = None
        self.partial = b""
        self.state = factory()

    def update(self):
        try:
            stat = self.path.stat()
        except OSError:
            return self.state
        if stat.st_ino != self.inode or stat.st_size < self.offset:
            self.inode, self.offset, self.partial = stat.st_ino, 0, b""
            self.state = self.factory()
        if stat.st_size == self.offset:
            return self.state
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read(stat.st_size - self.offset)
        self.offset += len(chunk)
        data = self.partial + chunk
        lines = data.split(b"\n")
        self.partial = lines.pop()
        for raw in lines:
            text = ANSI.sub("", raw.decode("utf-8", errors="replace"))
            for piece in text.split("\r"):
                if piece:
                    self.state.feed(piece)
        return self.state


class MasterLogState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.seeds: list[str] = []
        self.gpus: list[str] = []
        self.slots = None
        self.phase = "starting"
        self.failed = False
        self.completed = False
        self.errors: list[str] = []
        self.started: dict[str, dict] = {}

    def feed(self, line: str):
        if line.startswith("["):
            return  # container output relayed with a [run|gpu] prefix
        if line.startswith("Seeds: "):
            self.reset()
            self.seeds = line[len("Seeds: "):].split()
        elif line.startswith("Available GPUs: "):
            self.gpus = line[len("Available GPUs: "):].split()
        elif line.startswith("Concurrent slots: "):
            match = re.match(r"Concurrent slots: (\d+)", line)
            self.slots = int(match.group(1)) if match else None
        elif line.startswith("Preparing ds004408 word alignment"):
            self.phase = "preparing"
        elif line.startswith("Launching ") and " experiments" in line:
            self.phase = "finetuning"
        elif line.startswith("START "):
            fields = dict(
                part.strip().split("=", 1) for part in line.split("|")[1:] if "=" in part
            )
            name = line.split("|")[0][len("START "):].strip()
            match = re.match(r"ds004408_(.+_seed\d+)_", name)
            if match:
                self.started[match.group(1)] = {
                    "gpu": fields.get("GPU"),
                    "embedding": fields.get("embedding"),
                }
        elif line.startswith("Generating combined four-way report"):
            self.phase = "report"
        elif "four-way pipeline failed" in line:
            self.failed = True
        elif line.startswith("ds004408 four-way fine-tuning completed"):
            self.completed = True
        if line.startswith("ERROR"):
            self.errors = (self.errors + [line.strip()])[-10:]


class RunLogState:
    def __init__(self):
        self.first_ts = None
        self.epoch_end_ts: list[float] = []
        self.num_epochs = None
        self.current_epoch = None
        self.last_ts = None
        self.last_ts_text = None
        self.early_stopped = False
        self.final_eval = False
        self.traceback = False
        self.last_error = None

    def feed(self, line: str):
        match = ASCTIME.match(line)
        if match:
            ts = _asctime_to_ts(match.group(1))
            self.last_ts = ts
            if self.first_ts is None:
                self.first_ts = ts
            if "Metrics history updated" in line:
                self.epoch_end_ts.append(ts)
            elif "Early stopping at epoch" in line:
                self.early_stopped = True
            elif "Loading best model for final evaluation" in line:
                self.final_eval = True
            return
        match = EPOCH_LINE.match(line)
        if match:
            self.current_epoch = int(match.group(1))
            self.num_epochs = int(match.group(2))
        elif line.startswith("Traceback (most recent call last)"):
            self.traceback = True
        elif self.traceback and re.match(r"^[A-Za-z_.]+(Error|Exception)\b", line):
            self.last_error = line.strip()[:300]


_scanners: dict[str, IncrementalScanner] = {}
_scanner_lock = threading.Lock()


def _scan(path: Path, factory):
    key = str(path)
    with _scanner_lock:
        scanner = _scanners.get(key)
        if scanner is None:
            scanner = _scanners[key] = IncrementalScanner(path, factory)
        return scanner.update()


# --------------------------------------------------------------------------- #
# Run discovery and status
# --------------------------------------------------------------------------- #

def list_runs() -> list[str]:
    base = CFG.root / CFG.results_base
    try:
        runs = [p for p in base.iterdir() if p.is_dir() and SAFE_ID.match(p.name)]
    except OSError:
        runs = []
    runs.sort(key=lambda p: p.name, reverse=True)
    return [p.name for p in runs]


def latest_run() -> str | None:
    runs = list_runs()
    try:
        pointer = (CFG.root / "ds004408_four_way.latest").read_text().strip()
        if pointer in runs:
            return pointer
    except OSError:
        pass
    return runs[0] if runs else None


def resolve_run(requested: str | None) -> str | None:
    if requested and SAFE_ID.match(requested) and requested in list_runs():
        return requested
    return latest_run()


def _tqdm_progress(log_path: Path) -> dict | None:
    tail = ANSI.sub("", _read_tail(log_path, 32_768))
    pieces = re.split(r"[\r\n]", tail)
    for piece in reversed(pieces):
        match = TQDM.search(piece)
        if match:
            return {
                "stage": match.group(1).strip(),
                "percent": int(match.group(2)),
                "done": int(match.group(3)),
                "total": int(match.group(4)),
                "elapsed": match.group(5),
                "remaining": match.group(6),
            }
    return None


def _read_epochs(path: Path) -> list[dict]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        rows.append({k: _num(v) for k, v in row.items() if _num(v) is not None or k == "epoch"})
    return rows


def job_status(run_id: str, label: str, seed: str, master: MasterLogState,
               tsv_rows: dict, pipeline_over: bool, now: float) -> dict:
    run_key = f"{label}_seed{seed}"
    save_dir = CFG.root / CFG.logs_base / run_id / label / f"seed{seed}"
    log_path = save_dir / "stdout_stderr.log"
    epochs_path = save_dir / "epoch_metrics.jsonl"
    log_state = _scan(log_path, RunLogState) if log_path.exists() else RunLogState()
    epochs = _read_epochs(epochs_path)
    final = _read_json(save_dir / "final_results.json")
    tsv = tsv_rows.get(run_key)
    started_info = master.started.get(run_key, {})
    log_mtime = _mtime(log_path)

    num_epochs = log_state.num_epochs or CFG.default_epochs
    last = epochs[-1] if epochs else {}
    epoch_done = int(last.get("epoch") or 0)
    patience = int(last.get("training/patience_counter") or 0)
    early = bool(last.get("training/early_stopped")) or log_state.early_stopped

    if tsv:
        status = tsv.get("status", "UNKNOWN")
    elif run_key in master.started or log_mtime is not None:
        idle_min = (now - log_mtime) / 60 if log_mtime else None
        if idle_min is not None and idle_min > CFG.stale_minutes:
            status = "STALLED"
        else:
            status = "RUNNING"
    else:
        status = "NOT_RUN" if pipeline_over else "QUEUED"

    started_ts = _iso_to_ts(tsv.get("started_at")) if tsv else None
    if started_ts is None:
        started_ts = _mtime(save_dir / "config_resolved.yaml")
    finished_ts = _iso_to_ts(tsv.get("finished_at")) if tsv else None

    durations = [b - a for a, b in zip(log_state.epoch_end_ts, log_state.epoch_end_ts[1:])]
    if log_state.epoch_end_ts and log_state.first_ts is not None:
        durations.insert(0, log_state.epoch_end_ts[0] - log_state.first_ts)
    # The first epoch also pays for data loading; prefer steady-state epochs.
    steady = durations[1:] if len(durations) > 2 else durations
    epoch_seconds = sum(steady) / len(steady) if steady else None

    eta = None
    if status == "RUNNING" and epoch_seconds and not early:
        remaining = max(0, num_epochs - epoch_done)
        since_last = now - (_mtime(epochs_path) or now)
        eta = max(0.0, remaining * epoch_seconds - since_last)

    if status in ("COMPLETED", "FAILED", "NOT_RUN", "QUEUED"):
        phase = None
    elif log_state.final_eval:
        phase = {"stage": "Final test evaluation"}
    else:
        phase = _tqdm_progress(log_path) if log_path.exists() else None

    return {
        "key": run_key,
        "label": label,
        "model": MODEL_NAMES.get(label, label),
        "seed": seed,
        "gpu": (tsv or {}).get("gpu") or started_info.get("gpu"),
        "embedding": (tsv or {}).get("embedding_id") or started_info.get("embedding")
        or JOB_EMBEDDING_IDS.get(label),
        "status": status,
        "exit_code": (tsv or {}).get("exit_code"),
        "epoch": epoch_done,
        "num_epochs": num_epochs,
        "current_epoch": log_state.current_epoch,
        "patience": patience,
        "patience_max": CFG.default_patience,
        "early_stopped": early,
        "train_loss": last.get("train/loss"),
        "val_primary": last.get("val/primary_metric_value"),
        "val_best": last.get("val/best_primary_metric"),
        "val_best_epoch": last.get("val/best_epoch"),
        "phase": phase,
        "started_ts": started_ts,
        "finished_ts": finished_ts,
        "last_output_ts": log_mtime,
        "epoch_seconds": epoch_seconds,
        "eta_seconds": eta,
        "error": SECRETS.sub("[redacted]", log_state.last_error) if log_state.last_error else None,
        "test": {k: _num(v) for k, v in (final.get("test_metrics") or {}).items()},
        "history": [
            {
                "epoch": row.get("epoch"),
                "val_primary": row.get("val/primary_metric_value"),
                "val_b50": row.get("val/balanced_top10_accuracy_retrieval50"),
                "train_loss": row.get("train/loss"),
                "val_loss": row.get("val/loss"),
            }
            for row in epochs
        ],
    }


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def run_status(run_id: str) -> dict:
    now = time.time()
    results_dir = CFG.root / CFG.results_base / run_id
    master_path = CFG.root / CFG.master_log_pattern.format(run_id=run_id)
    master = _scan(master_path, MasterLogState) if master_path.exists() else MasterLogState()

    tsv_rows: dict[str, dict] = {}
    prep_row = None
    for row in _read_tsv(results_dir / "runs.tsv"):
        if row.get("label") == "prepare_word_aligned":
            prep_row = row
        else:
            tsv_rows[f"{row.get('label')}_seed{row.get('seed')}"] = row

    completed_marker = (results_dir / "COMPLETED").exists()
    master_mtime = _mtime(master_path)

    seeds = list(master.seeds)
    for key in list(tsv_rows) + list(master.started):
        seed = key.rsplit("_seed", 1)[-1]
        if seed not in seeds:
            seeds.append(seed)
    pipeline_over = completed_marker or master.failed

    jobs = [
        job_status(run_id, label, seed, master, tsv_rows, pipeline_over, now)
        for seed in seeds
        for label in JOB_LABELS
    ]
    any_fresh = any(
        j["last_output_ts"] and now - j["last_output_ts"] < CFG.stale_minutes * 60 for j in jobs
    )
    idle_min = (now - master_mtime) / 60 if master_mtime else None

    if completed_marker or master.completed:
        state, phase = "COMPLETED", "done"
    elif master.failed:
        state, phase = "FAILED", master.phase
    elif master_mtime is None:
        state, phase = "UNKNOWN", "no master log yet"
    elif idle_min is not None and idle_min > CFG.stale_minutes and not any_fresh:
        state, phase = "STALLED", master.phase
    else:
        state, phase = "RUNNING", master.phase

    counts: dict[str, int] = {}
    for job in jobs:
        counts[job["status"]] = counts.get(job["status"], 0) + 1

    progress = 0.0
    for job in jobs:
        if job["status"] == "COMPLETED":
            progress += 1.0
        elif job["num_epochs"]:
            progress += min(job["epoch"] / job["num_epochs"], 1.0) * 0.98
    progress = progress / len(jobs) if jobs else 0.0

    # Rough pipeline ETA: running batch + remaining batches at the mean
    # observed job duration (early stopping makes this an upper bound).
    eta = None
    done_durations = [
        j["finished_ts"] - j["started_ts"]
        for j in jobs
        if j["status"] == "COMPLETED" and j["finished_ts"] and j["started_ts"]
    ]
    running_etas = [j["eta_seconds"] for j in jobs if j["eta_seconds"] is not None]
    queued = counts.get("QUEUED", 0)
    slots = master.slots or max(1, counts.get("RUNNING", 0))
    if state == "RUNNING" and phase == "finetuning":
        per_job = None
        if done_durations:
            per_job = sum(done_durations) / len(done_durations)
        elif running_etas:
            per_job = max(
                (j["epoch_seconds"] or 0) * j["num_epochs"] for j in jobs if j["eta_seconds"]
            )
        if running_etas or per_job:
            eta = (max(running_etas) if running_etas else 0) + (
                math.ceil(queued / slots) * per_job if per_job and queued else 0
            )

    by_model = []
    for label in JOB_LABELS:
        finished = [j for j in jobs if j["label"] == label and j["test"]]
        entry = {"label": label, "model": MODEL_NAMES[label], "n": len(finished)}
        for metric in PRIMARY_TEST_METRICS:
            values = [j["test"][metric] for j in finished if j["test"].get(metric) is not None]
            entry[metric] = _mean_std(values)
        by_model.append(entry)

    report = None
    summary_csv = results_dir / "megxl_paper_metrics_summary.csv"
    if summary_csv.exists():
        figures = sorted(p.name for p in results_dir.glob("megxl_figure*.png"))
        report = {
            "summary": _read_csv(summary_csv),
            "welch": _read_csv(results_dir / "megxl_pairwise_welch_tests.csv"),
            "figures": figures,
        }

    start_candidates = [j["started_ts"] for j in jobs if j["started_ts"]]
    prep_started = _iso_to_ts(prep_row.get("started_at")) if prep_row else None
    if prep_started:
        start_candidates.append(prep_started)

    return {
        "run_id": run_id,
        "runs": list_runs(),
        "now": now,
        "state": state,
        "phase": phase,
        "seeds": seeds,
        "gpus": master.gpus,
        "slots": master.slots,
        "prep_done": prep_row is not None,
        "started_ts": min(start_candidates) if start_candidates else None,
        "last_activity_ts": max(
            [t for t in [master_mtime] + [j["last_output_ts"] for j in jobs] if t] or [0]
        ) or None,
        "progress": progress,
        "eta_seconds": eta,
        "counts": counts,
        "errors": master.errors,
        "jobs": jobs,
        "by_model": by_model,
        "report": report,
        "config": {
            "title": CFG.title,
            "gpu_dashboard_url": CFG.gpu_dashboard_url,
            "show_logs": CFG.show_logs,
            "stale_minutes": CFG.stale_minutes,
        },
    }


_gpu_cache: dict = {"ts": 0.0, "data": None}


def gpu_info() -> list[dict] | None:
    if time.time() - _gpu_cache["ts"] < 5:
        return _gpu_cache["data"]
    data = None
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=4, stderr=subprocess.DEVNULL,
        ).decode()
        data = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                data.append({
                    "index": parts[0], "name": parts[1], "util": _num(parts[2]),
                    "mem_used": _num(parts[3]), "mem_total": _num(parts[4]),
                    "temp": _num(parts[5]),
                })
    except (OSError, subprocess.SubprocessError):
        data = None
    _gpu_cache.update(ts=time.time(), data=data)
    return data


def log_tail(run_id: str, job: str, lines: int) -> str:
    if job == "master":
        path = CFG.root / CFG.master_log_pattern.format(run_id=run_id)
    else:
        match = re.fullmatch(r"([a-z0-9_]+)_seed(\d+)", job)
        if not match or match.group(1) not in JOB_LABELS:
            return "Unknown job."
        path = (CFG.root / CFG.logs_base / run_id / match.group(1)
                / f"seed{match.group(2)}" / "stdout_stderr.log")
    if not path.exists():
        return "No log yet."
    text = ANSI.sub("", _read_tail(path, 512_000))
    truncated = (_mtime(path) is not None) and path.stat().st_size > 512_000
    # Collapse carriage-return progress bars the way a terminal would.
    rendered = [line.rsplit("\r", 1)[-1] if "\r" in line.rstrip("\r") else line.rstrip("\r")
                for line in text.split("\n")]
    rendered = [line for line in rendered[1 if truncated else 0:] if line.strip()]
    return SECRETS.sub("[redacted]", "\n".join(rendered[-lines:]))


def prometheus_metrics() -> str:
    run_id = latest_run()
    if not run_id:
        return "# no ds004408 four-way runs found\n"
    status = run_status(run_id)
    out = [
        "# HELP ds004408_pipeline_progress Fraction of planned fine-tuning work done.",
        "# TYPE ds004408_pipeline_progress gauge",
        f'ds004408_pipeline_progress{{run="{run_id}",state="{status["state"]}"}} '
        f'{status["progress"]:.4f}',
        "# TYPE ds004408_jobs gauge",
    ]
    for state, count in status["counts"].items():
        out.append(f'ds004408_jobs{{run="{run_id}",status="{state}"}} {count}')
    gauges = [("epoch", "epoch"), ("val_primary", "val_primary"),
              ("val_best", "val_best"), ("train_loss", "train_loss"),
              ("eta_seconds", "eta_seconds")]
    for name, field in gauges:
        out.append(f"# TYPE ds004408_job_{name} gauge")
        for job in status["jobs"]:
            value = job.get(field)
            if value is not None:
                out.append(
                    f'ds004408_job_{name}{{run="{run_id}",model="{job["label"]}",'
                    f'seed="{job["seed"]}"}} {value}'
                )
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    server_version = "ds004408-monitor"

    def log_message(self, fmt, *args):
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200, cache: str = "no-store"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; frame-ancestors *",
        )
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data, status: int = 200):
        self._send(json.dumps(data, allow_nan=False, default=str).encode(),
                   "application/json", status)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        path = url.path.rstrip("/") or "/"
        # Tolerate being mounted under a reverse-proxy prefix without stripping.
        for suffix in ("/api/status", "/api/log", "/api/gpu", "/figure", "/metrics", "/healthz"):
            if path.endswith(suffix):
                path = suffix
                break
        try:
            if path == "/healthz":
                self._send(b"ok\n", "text/plain")
            elif path == "/api/status":
                run_id = resolve_run((query.get("run") or [None])[0])
                if run_id is None:
                    self._json({"run_id": None, "runs": [], "config": {
                        "title": CFG.title, "gpu_dashboard_url": CFG.gpu_dashboard_url,
                        "show_logs": CFG.show_logs}})
                else:
                    self._json(run_status(run_id))
            elif path == "/api/gpu":
                self._json({"gpus": gpu_info()})
            elif path == "/api/log":
                if not CFG.show_logs:
                    self._send(b"Logs are disabled on this monitor.", "text/plain; charset=utf-8")
                    return
                run_id = resolve_run((query.get("run") or [None])[0])
                job = (query.get("job") or ["master"])[0]
                lines = min(max(int((query.get("lines") or ["200"])[0]), 10), 2000)
                text = log_tail(run_id, job, lines) if run_id else "No run."
                self._send(text.encode(), "text/plain; charset=utf-8")
            elif path == "/figure":
                run_id = resolve_run((query.get("run") or [None])[0])
                name = (query.get("name") or [""])[0]
                if not run_id or not re.fullmatch(r"megxl_figure[A-Za-z0-9_]*\.png", name):
                    self._send(b"not found", "text/plain", 404)
                    return
                figure = CFG.root / CFG.results_base / run_id / name
                if not figure.is_file():
                    self._send(b"not found", "text/plain", 404)
                    return
                self._send(figure.read_bytes(), "image/png", cache="max-age=300")
            elif path == "/metrics":
                self._send(prometheus_metrics().encode(), "text/plain; version=0.0.4")
            elif path in ("/", "/index.html"):
                run_id = resolve_run((query.get("run") or [None])[0])
                initial = json.dumps(run_status(run_id), allow_nan=False, default=str) if run_id else "null"
                page = INDEX_HTML.replace("__INITIAL_STATE__", initial.replace("</", "<\\/"))
                self._send(page.encode(), "text/html; charset=utf-8")
            else:
                self._send(b"not found", "text/plain", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # keep the monitor alive on malformed files
            self._json({"error": type(exc).__name__, "detail": str(exc)[:300]}, 500)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ds004408 Run Monitor</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --wash: rgba(11,11,11,0.04);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --good: #0ca30c; --good-text: #006300; --warning: #fab219; --critical: #d03b3b; --running: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --wash: rgba(255,255,255,0.05);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    --good-text: #0ca30c; --running: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --wash: rgba(255,255,255,0.05);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --good-text: #0ca30c; --running: #3987e5;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 20px 16px 48px; }
header { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: flex-end; justify-content: space-between; }
h1 { font-size: 20px; margin: 0; font-weight: 650; letter-spacing: -0.01em; }
h2 { font-size: 15px; margin: 0 0 12px; font-weight: 620; }
.sub { color: var(--ink-2); font-size: 13px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
select, button { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 5px 10px; }
button { cursor: pointer; }
button:hover, select:hover { background: var(--wash); }
a { color: var(--running); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-top: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; }
.tile .k { color: var(--ink-2); font-size: 12px; }
.tile .v { font-size: 22px; font-weight: 620; margin-top: 2px; }
.tile .d { color: var(--muted); font-size: 12px; }
.badge { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; white-space: nowrap; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: none; }
.st-COMPLETED .dot { background: var(--good); }
.st-RUNNING .dot { background: var(--running); animation: pulse 1.6s ease-in-out infinite; }
.st-FAILED .dot { background: var(--critical); }
.st-STALLED .dot { background: var(--warning); }
.st-QUEUED .dot, .st-NOT_RUN .dot, .st-UNKNOWN .dot { background: transparent; border: 1.5px solid var(--muted); }
@keyframes pulse { 50% { opacity: .35; } }
@media (prefers-reduced-motion: reduce) { .st-RUNNING .dot { animation: none; } }
.bar { height: 6px; background: var(--grid); border-radius: 3px; overflow: hidden; min-width: 80px; }
.bar > span { display: block; height: 100%; background: var(--running); border-radius: 3px; }
.bar.big { height: 10px; border-radius: 5px; }
.stages { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.stage { padding: 4px 10px; border-radius: 999px; border: 1px solid var(--border); color: var(--muted); font-size: 12px; }
.stage.done { color: var(--good-text); }
.stage.active { color: var(--ink); border-color: var(--running); font-weight: 600; }
.table-wrap { overflow-x: auto; margin: 0 -16px; padding: 0 16px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { text-align: left; font-weight: 560; color: var(--ink-2); padding: 6px 8px; border-bottom: 1px solid var(--axis); white-space: nowrap; }
td { padding: 7px 8px; border-bottom: 1px solid var(--grid); vertical-align: middle; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tr.job { cursor: pointer; }
tr.job:hover { background: var(--wash); }
tr.job.selected { background: var(--wash); box-shadow: inset 3px 0 0 var(--running); }
.key { display: inline-block; width: 16px; height: 0; border-top: 2px solid; vertical-align: middle; margin-right: 6px; }
.muted { color: var(--muted); }
.small { font-size: 12px; }
.err { color: var(--critical); font-size: 12px; }
.chart-head { display: flex; flex-wrap: wrap; gap: 8px 16px; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--ink-2); }
.chart { position: relative; }
.chart svg { display: block; width: 100%; height: 300px; overflow: visible; }
.tooltip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.12); min-width: 170px; display: none; z-index: 2; }
.tooltip .row { display: flex; gap: 8px; align-items: center; white-space: nowrap; }
.tooltip .row b { font-variant-numeric: tabular-nums; min-width: 48px; }
pre.log { background: var(--page); border: 1px solid var(--border); border-radius: 8px; padding: 10px; max-height: 420px;
  overflow: auto; font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; word-break: break-word; margin: 0; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.gpu { display: grid; grid-template-columns: 56px 1fr 60px; gap: 4px 10px; align-items: center; font-size: 12px; }
.figs { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
.figs img { width: 100%; border-radius: 8px; border: 1px solid var(--border); background: #fff; }
.stale { opacity: .6; transition: opacity .2s; }
footer { margin-top: 24px; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<main id="app">
  <header>
    <div>
      <h1 id="title">ds004408 four-way fine-tuning</h1>
      <div class="sub" id="subtitle">Loading…</div>
    </div>
    <div class="controls">
      <label class="sub" for="run">Run</label>
      <select id="run"></select>
      <button id="theme" type="button" title="Toggle light/dark">◐</button>
      <a id="gpu-link" class="sub" target="_blank" rel="noopener" hidden>GPU dashboard ↗</a>
    </div>
  </header>

  <section class="tiles" id="tiles"></section>

  <section class="card">
    <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center">
      <h2 style="margin:0">Pipeline</h2>
      <span class="sub" id="progress-label"></span>
    </div>
    <div class="bar big" style="margin-top:10px"><span id="progress-bar" style="width:0"></span></div>
    <div class="stages" id="stages"></div>
    <div id="errors"></div>
  </section>

  <section class="card">
    <h2>Fine-tuning jobs</h2>
    <div class="table-wrap"><table id="jobs"></table></div>
    <div class="sub small" style="margin-top:8px">Val metric = balanced top-10 accuracy with a 250-word retrieval set (chance 4%). Click a row to follow its log.</div>
  </section>

  <section class="card">
    <div class="chart-head">
      <h2 style="margin:0" id="chart-title">Validation curve</h2>
      <select id="metric" aria-label="Metric">
        <option value="val_primary">Val balanced top-10 @250</option>
        <option value="val_b50">Val balanced top-10 @50</option>
        <option value="train_loss">Train loss</option>
        <option value="val_loss">Val loss</option>
      </select>
    </div>
    <div class="legend" id="legend"></div>
    <div class="chart" id="chart"><svg id="svg" role="img" aria-label="Metric by epoch"></svg><div class="tooltip" id="tip"></div></div>
  </section>

  <div class="grid2">
    <section class="card">
      <h2>Test results so far</h2>
      <div class="table-wrap"><table id="results"></table></div>
      <div class="sub small" style="margin-top:8px">Mean ± SD over finished seeds (best-val checkpoint on the held-out test split). Chance: 4% @250, 20% @50.</div>
    </section>
    <section class="card" id="gpu-card" hidden>
      <h2>GPUs</h2>
      <div class="gpu" id="gpus"></div>
    </section>
  </div>

  <section class="card" id="report-card" hidden>
    <h2>Final comparison report</h2>
    <div class="figs" id="figs"></div>
    <div class="table-wrap" style="margin-top:12px"><table id="summary"></table></div>
    <div class="table-wrap" style="margin-top:12px"><table id="welch"></table></div>
  </section>

  <section class="card" id="log-card">
    <div class="chart-head">
      <h2 style="margin:0">Log · <span id="log-name">pipeline</span></h2>
      <div class="controls">
        <button type="button" id="log-master">Pipeline log</button>
        <label class="sub"><input type="checkbox" id="log-follow" checked> follow</label>
      </div>
    </div>
    <pre class="log" id="log">…</pre>
  </section>

  <footer id="footer"></footer>
</main>
<script id="initial-state" type="application/json">__INITIAL_STATE__</script>
<script>
"use strict";
const REFRESH_MS = 15000;
const SERIES = {random_init: "--s1", eeg_from_scratch: "--s2", megxl_eeg2: "--s3", megxl_eeg1: "--s4"};
const DASHES = ["", "6 4", "2 3", "10 3 2 3", "1 5"];
const $ = (id) => document.getElementById(id);
let state = null, selectedJob = null, currentRun = null;
try { selectedJob = localStorage.getItem("ds4408-job"); } catch (e) {}

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}
function svgEl(tag, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  return node;
}
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const pct = (v, d = 1) => v == null ? "–" : (v * 100).toFixed(d) + "%";
const fix = (v, d = 3) => v == null ? "–" : Number(v).toFixed(d);
function dur(s) {
  if (s == null || !isFinite(s)) return "–";
  s = Math.max(0, Math.round(s));
  const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  return `${m}m ${String(s % 60).padStart(2, "0")}s`;
}
function ago(ts) { return ts ? dur(Date.now() / 1000 - ts) + " ago" : "–"; }
function clock(ts) { return ts ? new Date(ts * 1000).toLocaleString([], {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"}) : "–"; }
const STATUS_TEXT = {COMPLETED: "Completed", RUNNING: "Running", FAILED: "Failed", STALLED: "No output", QUEUED: "Queued", NOT_RUN: "Not run", UNKNOWN: "Unknown"};
function badge(status) {
  return el("span", {class: `badge st-${status}`}, el("span", {class: "dot", "aria-hidden": "true"}), STATUS_TEXT[status] || status);
}
function metricKeyForModel(label) { return el("span", {class: "key", style: `border-color: var(${SERIES[label] || "--muted"})`}); }

function renderHeader(s) {
  $("title").textContent = s.config.title;
  document.title = "ds004408 Run Monitor" + (s.run_id ? " · " + s.state.toLowerCase() : "");
  const link = $("gpu-link");
  if (s.config.gpu_dashboard_url) { link.href = s.config.gpu_dashboard_url; link.hidden = false; }
  const sel = $("run");
  sel.replaceChildren(...(s.runs || []).map((r) => el("option", {value: r, selected: r === s.run_id ? "" : null}, r)));
  if (!s.run_id) { $("subtitle").textContent = "No runs found yet under results/ds004408_four_way/."; return; }
  $("subtitle").replaceChildren(badge(s.state), "  ·  Run ", s.run_id,
    "  ·  seeds ", (s.seeds || []).join(", ") || "–",
    "  ·  GPUs ", (s.gpus || []).join(", ") || "–",
    "  ·  last activity ", ago(s.last_activity_ts));
}

function renderTiles(s) {
  const c = s.counts || {}, total = s.jobs.length;
  const elapsed = s.started_ts ? (s.state === "RUNNING" || s.state === "STALLED" ? s.now : Math.max(...s.jobs.map((j) => j.finished_ts || 0), s.started_ts)) - s.started_ts : null;
  const best = s.jobs.filter((j) => j.val_best != null).sort((a, b) => b.val_best - a.val_best)[0];
  const tiles = [
    ["Jobs completed", `${c.COMPLETED || 0} / ${total}`, `${c.RUNNING || 0} running · ${c.QUEUED || 0} queued` + (c.FAILED ? ` · ${c.FAILED} failed` : "")],
    ["Overall progress", pct(s.progress, 0), "epoch-weighted"],
    ["Elapsed", dur(elapsed), s.started_ts ? "since " + clock(s.started_ts) : ""],
    ["Remaining (upper bound)", s.eta_seconds != null ? "≈ " + dur(s.eta_seconds) : "–", s.eta_seconds != null ? "done ≈ " + clock(s.now + s.eta_seconds) : "early stopping shortens this"],
    ["Best val so far", best ? pct(best.val_best) : "–", best ? `${best.model} · seed ${best.seed}` : ""],
  ];
  $("tiles").replaceChildren(...tiles.map(([k, v, d]) => el("div", {class: "tile"}, el("div", {class: "k"}, k), el("div", {class: "v"}, v), el("div", {class: "d"}, d))));
}

function renderPipeline(s) {
  $("progress-bar").style.width = (s.progress * 100).toFixed(1) + "%";
  $("progress-label").textContent = `${pct(s.progress, 1)} · phase: ${s.phase}`;
  const order = ["preparing", "finetuning", "report", "done"];
  const idx = order.indexOf(s.phase);
  const names = ["Word alignment", "Fine-tuning", "Comparison report", "Done"];
  $("stages").replaceChildren(...names.map((n, i) => {
    let cls = "stage";
    if (s.state === "COMPLETED" || i < idx || (i === 0 && s.prep_done && idx < 1)) cls += " done";
    else if (i === idx) cls += " active";
    return el("span", {class: cls}, (cls.includes("done") ? "✓ " : "") + n);
  }));
  $("errors").replaceChildren(...(s.errors || []).slice(-5).map((e) => el("div", {class: "err"}, "✕ " + e)));
}

function renderJobs(s) {
  const head = el("tr", {}, ...["Model", "Seed", "GPU", "Status", "Epoch", "Now", "Train loss", "Val @250", "Best val", "Test @250", "ETA"].map((h, i) =>
    el("th", {class: i >= 4 && i !== 5 ? "num" : null}, h)));
  const rows = s.jobs.map((j) => {
    const frac = j.status === "COMPLETED" ? 1 : j.num_epochs ? j.epoch / j.num_epochs : 0;
    let now = "–";
    if (j.status === "STALLED") now = `no output for ${ago(j.last_output_ts).replace(" ago", "")}`;
    else if (j.status === "COMPLETED") now = j.early_stopped ? "early-stopped" : "finished";
    else if (j.status === "FAILED") now = "exit " + (j.exit_code ?? "?");
    else if (j.phase) now = j.phase.percent != null ? `${j.phase.stage} ${j.phase.percent}% (${j.phase.remaining} left)` : j.phase.stage;
    const tr = el("tr", {class: "job" + (j.key === selectedJob ? " selected" : ""), tabindex: "0",
      onclick: () => selectJob(j.key), onkeydown: (e) => { if (e.key === "Enter") selectJob(j.key); }},
      el("td", {style: "min-width:190px"}, metricKeyForModel(j.label), j.model, j.embedding ? el("div", {class: "muted small", style: "padding-left:22px"}, `embedding row ${j.embedding}`) : null),
      el("td", {}, j.seed), el("td", {}, j.gpu ?? "–"),
      el("td", {}, badge(j.status), j.error ? el("div", {class: "err", title: j.error, style: "max-width:190px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis"}, j.error) : null),
      el("td", {class: "num", title: "Epochs without improvement / early-stopping patience"}, el("div", {}, `${j.epoch} / ${j.num_epochs}`),
        el("div", {class: "bar", style: "margin-top:4px"}, el("span", {style: `width:${(frac * 100).toFixed(1)}%`})),
        j.epoch && j.status !== "COMPLETED" ? el("div", {class: "muted small"}, `patience ${j.patience}/${j.patience_max}`) : null),
      el("td", {class: "small", style: "min-width:130px"}, now),
      el("td", {class: "num"}, fix(j.train_loss, 1)),
      el("td", {class: "num"}, pct(j.val_primary)),
      el("td", {class: "num"}, j.val_best != null ? `${pct(j.val_best)} (ep ${j.val_best_epoch})` : "–"),
      el("td", {class: "num"}, pct(j.test.balanced_top10_accuracy_retrieval250)),
      el("td", {class: "num"}, j.eta_seconds != null ? dur(j.eta_seconds)
        : (j.status === "COMPLETED" || j.status === "FAILED") && j.finished_ts && j.started_ts ? el("span", {class: "muted", title: "total run time"}, dur(j.finished_ts - j.started_ts)) : "–"));
    return tr;
  });
  $("jobs").replaceChildren(el("thead", {}, head), el("tbody", {}, rows));
}

function renderResults(s) {
  const head = el("tr", {}, el("th", {}, "Model"), el("th", {class: "num"}, "Seeds done"), el("th", {class: "num"}, "Test @250"), el("th", {class: "num"}, "Test @50"));
  const fmt = (ms) => ms && ms[0] != null ? pct(ms[0]) + (ms[1] != null ? " ± " + pct(ms[1]) : "") : "–";
  const rows = s.by_model.map((m) => el("tr", {}, el("td", {style: "white-space:nowrap"}, metricKeyForModel(m.label), m.model),
    el("td", {class: "num"}, `${m.n} / ${s.seeds.length}`),
    el("td", {class: "num"}, fmt(m.balanced_top10_accuracy_retrieval250)),
    el("td", {class: "num"}, fmt(m.balanced_top10_accuracy_retrieval50))));
  $("results").replaceChildren(el("thead", {}, head), el("tbody", {}, rows));
}

function genericTable(rows, preferred) {
  if (!rows || !rows.length) return [];
  let cols = Object.keys(rows[0]);
  if (preferred) cols = preferred.filter((c) => cols.includes(c));
  const isNum = (v) => v !== "" && !isNaN(Number(v));
  return [el("thead", {}, el("tr", {}, cols.map((c) => el("th", {class: isNum(rows[0][c]) ? "num" : null}, c)))),
    el("tbody", {}, rows.map((r) => el("tr", {}, cols.map((c) => el("td", {class: isNum(r[c]) ? "num" : null},
      isNum(r[c]) && String(r[c]).includes(".") ? Number(r[c]).toPrecision(4) : r[c])))))];
}

function renderReport(s) {
  const card = $("report-card");
  if (!s.report) { card.hidden = true; return; }
  card.hidden = false;
  $("figs").replaceChildren(...s.report.figures.map((f) => el("a", {href: `figure?run=${encodeURIComponent(s.run_id)}&name=${encodeURIComponent(f)}`, target: "_blank", rel: "noopener"},
    el("img", {src: `figure?run=${encodeURIComponent(s.run_id)}&name=${encodeURIComponent(f)}`, alt: f, loading: "lazy"}))));
  $("summary").replaceChildren(...genericTable(s.report.summary, ["model_display", "retrieval_size", "n_seeds", "balanced_mean", "balanced_std", "balanced_ci95_low", "balanced_ci95_high", "chance_accuracy", "balanced_mean_x_chance"]));
  $("welch").replaceChildren(...genericTable(s.report.welch));
}

// ---- line chart: one series per job, colour = model, dash = seed ----
function renderChart(s) {
  const metric = $("metric").value;
  const isPct = metric.startsWith("val_b") || metric === "val_primary";
  $("chart-title").textContent = $("metric").selectedOptions[0].textContent + " by epoch";
  const seeds = s.seeds || [];
  const series = s.jobs.filter((j) => j.history.some((h) => h[metric] != null)).map((j) => ({
    job: j, color: css(SERIES[j.label] || "--muted"), dash: DASHES[seeds.indexOf(j.seed) % DASHES.length] || "",
    pts: j.history.filter((h) => h[metric] != null && h.epoch != null).map((h) => [h.epoch, h[metric]]),
  }));
  const legendItems = Object.keys(SERIES).map((label) => el("span", {}, metricKeyForModel(label), s.jobs.find((j) => j.label === label)?.model || label));
  if (seeds.length > 1) legendItems.push(el("span", {class: "muted"}, "line style = seed: " + seeds.map((sd, i) => `${sd} ${["solid", "dashed", "dotted", "dash-dot", "sparse"][i % 5]}`).join(", ")));
  $("legend").replaceChildren(...legendItems);

  const svg = $("svg");
  const W = svg.clientWidth || 800, H = 300, m = {l: 52, r: 16, t: 10, b: 30};
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.replaceChildren();
  if (!series.length) {
    svg.append(Object.assign(svgEl("text", {x: W / 2, y: H / 2, "text-anchor": "middle", fill: css("--muted"), "font-size": 13}), {textContent: "No finished epochs yet — curves appear after epoch 1."}));
    return;
  }
  const maxEpoch = Math.max(...s.jobs.map((j) => j.num_epochs || 1), ...series.flatMap((d) => d.pts.map((p) => p[0])));
  const vals = series.flatMap((d) => d.pts.map((p) => p[1]));
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (isPct) { lo = Math.min(0, lo); hi = Math.max(hi * 1.1, 0.05); } else { const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 1; lo -= pad; hi += pad; }
  const x = (e) => m.l + (e - 1) / Math.max(1, maxEpoch - 1) * (W - m.l - m.r);
  const y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
  const rawStep = (hi - lo) / 5, mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((v) => v >= rawStep);
  lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
  const ticks = Math.round((hi - lo) / step), fmtTick = (v) => isPct ? (v * 100).toFixed(Number.isInteger(Math.round(step * 1e4) / 100) ? 0 : 1) + "%" : Math.abs(v) >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(v < 10 ? 2 : 0);
  for (let i = 0; i <= ticks; i++) {
    const v = lo + (hi - lo) * i / ticks, yy = y(v);
    svg.append(svgEl("line", {x1: m.l, x2: W - m.r, y1: yy, y2: yy, stroke: css(i === 0 ? "--axis" : "--grid"), "stroke-width": 1}));
    svg.append(Object.assign(svgEl("text", {x: m.l - 8, y: yy + 4, "text-anchor": "end", fill: css("--muted"), "font-size": 11, style: "font-variant-numeric: tabular-nums"}), {textContent: fmtTick(v)}));
  }
  const xStep = maxEpoch > 30 ? 10 : maxEpoch > 12 ? 5 : 1;
  for (let e = 1; e <= maxEpoch; e += (e === 1 && xStep > 1 ? xStep - 1 : xStep)) {
    svg.append(Object.assign(svgEl("text", {x: x(e), y: H - 10, "text-anchor": "middle", fill: css("--muted"), "font-size": 11}), {textContent: e}));
  }
  if (metric === "val_primary") {
    const yc = y(0.04);
    svg.append(svgEl("line", {x1: m.l, x2: W - m.r, y1: yc, y2: yc, stroke: css("--muted"), "stroke-dasharray": "3 4", "stroke-width": 1}));
    svg.append(Object.assign(svgEl("text", {x: W - m.r, y: yc - 5, "text-anchor": "end", fill: css("--muted"), "font-size": 11}), {textContent: "chance 4%"}));
  }
  for (const d of series) {
    const path = d.pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
    const faded = selectedJob && selectedJob !== d.job.key;
    svg.append(svgEl("path", {d: path, fill: "none", stroke: d.color, "stroke-width": d.job.key === selectedJob ? 2.5 : 2, "stroke-dasharray": d.dash, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: faded ? 0.35 : 1}));
    const lp = d.pts[d.pts.length - 1];
    svg.append(svgEl("circle", {cx: x(lp[0]), cy: y(lp[1]), r: 4, fill: d.color, stroke: css("--surface"), "stroke-width": 2, opacity: faded ? 0.35 : 1}));
  }
  const cross = svgEl("line", {y1: m.t, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden"});
  svg.append(cross);
  const hit = svgEl("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent"});
  svg.append(hit);
  const tip = $("tip");
  const move = (ev) => {
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) * W / r.width;
    const e = Math.min(maxEpoch, Math.max(1, Math.round(1 + (px - m.l) / (W - m.l - m.r) * (maxEpoch - 1))));
    cross.setAttribute("x1", x(e)); cross.setAttribute("x2", x(e)); cross.setAttribute("visibility", "visible");
    const rows = series.map((d) => [d, d.pts.find((p) => p[0] === e)]).filter(([, p]) => p).sort((a, b) => b[1][1] - a[1][1]);
    tip.replaceChildren(el("div", {class: "muted", style: "margin-bottom:4px"}, `Epoch ${e}`),
      ...(rows.length ? rows.map(([d, p]) => el("div", {class: "row"},
        el("span", {class: "key", style: `border-color:${d.color};border-top-style:${d.dash ? "dashed" : "solid"}`}),
        el("b", {}, isPct ? pct(p[1]) : fix(p[1], 2)), el("span", {class: "muted"}, `${d.job.model} · s${d.job.seed}`))) : [el("div", {class: "muted"}, "no data")]));
    tip.style.display = "block";
    const left = (x(e) / W) * r.width;
    tip.style.left = Math.min(left + 12, r.width - tip.offsetWidth - 4) + "px";
    tip.style.top = "8px";
  };
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
}

async function refreshGpu() {
  try {
    const r = await fetch("api/gpu", {cache: "no-store"});
    const {gpus} = await r.json();
    if (!gpus || !gpus.length) { $("gpu-card").hidden = true; return; }
    $("gpu-card").hidden = false;
    const busy = new Set((state?.jobs || []).filter((j) => j.status === "RUNNING").map((j) => String(j.gpu)));
    $("gpus").replaceChildren(...gpus.flatMap((g) => [
      el("span", {}, `GPU ${g.index}`, busy.has(g.index) ? el("span", {class: "muted"}, " ●") : null),
      el("div", {}, el("div", {class: "bar", title: `utilization ${g.util}%`}, el("span", {style: `width:${g.util || 0}%`})),
        el("div", {class: "bar", style: "margin-top:3px", title: `memory ${g.mem_used}/${g.mem_total} MiB`}, el("span", {style: `width:${g.mem_total ? 100 * g.mem_used / g.mem_total : 0}%;background:var(--muted)`}))),
      el("span", {class: "num muted", style: "text-align:right"}, `${g.util ?? "–"}% · ${g.mem_total ? Math.round(g.mem_used / 1024) + "G" : ""}`),
    ]), el("span", {class: "muted small", style: "grid-column:1/-1"}, "Top bar: utilization · bottom bar: memory · ● hosts a job from this run"));
  } catch (e) { $("gpu-card").hidden = true; }
}

async function refreshLog() {
  if (!state || !state.config.show_logs) { $("log-card").hidden = true; return; }
  const job = selectedJob || "master";
  $("log-name").textContent = job === "master" ? "pipeline" : job;
  try {
    const r = await fetch(`api/log?run=${encodeURIComponent(currentRun || "")}&job=${encodeURIComponent(job)}&lines=250`, {cache: "no-store"});
    const pre = $("log");
    pre.textContent = await r.text();
    if ($("log-follow").checked) pre.scrollTop = pre.scrollHeight;
  } catch (e) {}
}

function selectJob(key) {
  selectedJob = key;
  try { key ? localStorage.setItem("ds4408-job", key) : localStorage.removeItem("ds4408-job"); } catch (e) {}
  if (state) { renderJobs(state); renderChart(state); }
  refreshLog();
}

function render(s) {
  state = s;
  currentRun = s.run_id;
  renderHeader(s);
  if (s.run_id) { renderTiles(s); renderPipeline(s); renderJobs(s); renderChart(s); renderResults(s); renderReport(s); }
  $("footer").textContent = `Updated ${new Date().toLocaleTimeString()} · refreshes every ${REFRESH_MS / 1000}s · read-only view`;
}

async function refresh() {
  document.body.classList.add("stale");
  try {
    const r = await fetch(`api/status${currentRun ? "?run=" + encodeURIComponent(currentRun) : ""}`, {cache: "no-store"});
    const s = await r.json();
    if (s.error) throw new Error(s.detail || s.error);
    render(s);
  } catch (e) {
    $("footer").textContent = "Update failed: " + e.message + " — retrying.";
  }
  document.body.classList.remove("stale");
  refreshLog();
  refreshGpu();
}

$("run").addEventListener("change", (e) => { currentRun = e.target.value; selectJob(null); refresh(); });
$("metric").addEventListener("change", () => state && renderChart(state));
$("log-master").addEventListener("click", () => selectJob(null));
$("theme").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("ds4408-theme", document.documentElement.dataset.theme); } catch (e) {}
  if (state) renderChart(state);
});
try { const t = localStorage.getItem("ds4408-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
new ResizeObserver(() => state && state.run_id && renderChart(state)).observe($("chart"));
try {
  const initial = JSON.parse($("initial-state").textContent);
  if (initial) { render(initial); refreshLog(); refreshGpu(); } else refresh();
} catch (e) { refresh(); }
setInterval(refresh, REFRESH_MS);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=os.environ.get("EEG_XL_ROOT", "/workspace"))
    parser.add_argument("--host", default=os.environ.get("MONITOR_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MONITOR_PORT", "8080")))
    args = parser.parse_args()

    CFG.root = Path(args.root).resolve()
    CFG.results_base = os.environ.get("RESULTS_BASE", CFG.results_base)
    CFG.logs_base = os.environ.get("LOGS_BASE", CFG.logs_base)
    CFG.master_log_pattern = os.environ.get("MASTER_LOG_PATTERN", CFG.master_log_pattern)
    CFG.stale_minutes = float(os.environ.get("STALE_MINUTES", CFG.stale_minutes))
    CFG.default_epochs = int(os.environ.get("NUM_EPOCHS", CFG.default_epochs))
    CFG.default_patience = int(os.environ.get("PATIENCE", CFG.default_patience))
    CFG.show_logs = os.environ.get("MONITOR_SHOW_LOGS", "1").lower() in ("1", "true", "yes")
    CFG.title = os.environ.get("MONITOR_TITLE", CFG.title)
    CFG.gpu_dashboard_url = os.environ.get("GPU_DASHBOARD_URL", "")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"ds004408 monitor on http://{args.host}:{args.port} (root={CFG.root})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
