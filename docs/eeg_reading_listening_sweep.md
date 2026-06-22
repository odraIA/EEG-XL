# Staged EEG reading → listening sweep

This launcher runs one independent experiment per GPU and automatically assigns the next queued experiment when a GPU becomes free. Every experiment has two consecutive stages:

1. train on continuous **reading** EEG (EEGDash NM000228 + ZuCo NR);
2. train on continuous **listening** EEG (ds004408 + ds007808 + SparrKULee), initialized from the best checkpoint produced in stage 1.

The existing Docker working directory and dataset mounts are preserved. All stages use:

```text
/workspace/datasets/...
./data/cache/eeg_preprocessed
./logs/eeg_reading_listening_training
./checkpoints/eeg_reading_listening_training
```

## Experiment count

The retained bands are:

| Band | Range | Fixed 50 Hz | Nyquist-aware |
|---|---:|---:|---:|
| alpha | 8–12 Hz | 50 Hz | — |
| beta | 13–24 Hz | 50 Hz | — |
| low-gamma | 30–45 Hz | 50 Hz | 100 Hz |
| full-band | 0.1–50 Hz | 50 Hz | 128 Hz |

The fixed-50-Hz versions of low-gamma and full-band are deliberate controls of the MEG-XL filter-then-resample setup. They do not preserve the complete requested frequency range after resampling, but they are kept to test whether that behavior contributes to performance. The Nyquist-aware counterparts retain the requested bands.

Each tokenizer/initialization pair therefore contains six profiles:

```text
alpha fixed50
beta fixed50
low-gamma fixed50
low-gamma Nyquist-aware
full-band fixed50
full-band Nyquist-aware
```

With two tokenizers and two initialization modes:

```text
6 profiles × 2 tokenizers × 2 initializations = 24 pipelines
48 training stages = 24 reading + 24 listening
24 final listening models
```

Default tokenizers:

```text
biocodec
brainomni_base
```

Default initialization modes:

```text
scratch
pretrained
```

## Run

From the repository root:

```bash
bash scripts/run_eeg_reading_then_listening_sweep.sh
```

Defaults:

- GPU workers: `0 1`;
- one training process per GPU;
- initial batch-size candidates: `16 12 8 6 4 2 1`;
- 50 epochs per stage, inherited from the existing configs;
- continue with the next queued pipeline after a failure.

The shared cache can be overridden explicitly:

```bash
EEG_CACHE_DIR=./data/cache/eeg_preprocessed \
  bash scripts/run_eeg_reading_then_listening_sweep.sh
```

## Automatic batch sizing

Before the first full run for a `(stage, band, sampling profile, tokenizer, GPU-memory size)` combination, the launcher executes a two-step probe from the largest candidate to the smallest. The first candidate that completes is cached in `batch_sizes.tsv`. If a real run raises an OOM, it is retried with the next smaller candidate.

To start conservatively at batch size 4 and only try smaller values:

```bash
EEG_BATCH_CANDIDATES="4 2 1" \
  bash scripts/run_eeg_reading_then_listening_sweep.sh
```

To disable probing and force batch size 4:

```bash
EEG_AUTO_BATCH=false EEG_DEFAULT_BATCH_SIZE=4 \
  bash scripts/run_eeg_reading_then_listening_sweep.sh
```

## Useful smoke tests

Generate the complete queue without launching Docker:

```bash
EEG_DRY_RUN=1 bash scripts/run_eeg_reading_then_listening_sweep.sh
```

Run only two pipelines, with ten steps per stage:

```bash
EEG_SWEEP_LIMIT=2 EEG_MAX_STEPS=10 EEG_BATCH_CANDIDATES="4 2 1" \
  bash scripts/run_eeg_reading_then_listening_sweep.sh
```

Use different GPU IDs:

```bash
EEG_GPUS="2 3" bash scripts/run_eeg_reading_then_listening_sweep.sh
```

## Results

Each launch creates:

```text
results/eeg_reading_listening_sweep/<timestamp>/
├── jobs.tsv
├── runs.tsv
├── batch_sizes.tsv
├── sweep_metadata.txt
├── final_results.txt
├── batch_probes/
└── stages/
```

The actual model outputs remain in:

```text
logs/eeg_reading_listening_training/<experiment>/
checkpoints/eeg_reading_listening_training/<experiment>/
```

A completed reading stage exposes `checkpoint_best.pt`, or `checkpoint_latest.pt` as fallback. That checkpoint is passed to the corresponding listening stage through `model.promoted_checkpoint`.
