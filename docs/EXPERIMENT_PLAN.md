# EEG-XL: band × sampling-rate × seed sweep — design doc

Status: draft. Everything under "Confirmed against the repo" is backed by
docs/legacy_guides/*.md or CITATION/pyproject already pulled from
github.com/odraIA/EEG-XL. Everything under "Needs confirmation" requires
re-reading files this session currently can't reach (GitHub connector
dropped) — see the TODOs in the launcher script.

## 1. Bands (Hz)

Source: Hollenstein et al., "Decoding EEG Brain Activity for Multi-Modal
Natural Language Processing" (`EEG_bands.pdf` in project knowledge), plus
the original MEG-XL preprocessing band (`alfons_meg_model.pdf`, Table 3).

| Band | l_freq | h_freq |
|---|---|---|
| theta | 4.0 | 8.0 |
| alpha | 8.5 | 13.0 |
| beta | 13.5 | 30.0 |
| gamma | 30.5 | 49.5 |
| broadband_megxl | 0.1 | 40.0 |

## 2. Sampling rates (Hz)

50, 100, 150, 200, 250

## 3. Nyquist-respecting combinations

A (band, fs) pair is valid iff `band.h_freq <= fs / 2`, with one deliberate
exception: `broadband_megxl @ 50 Hz` is kept as an **anchor run** — it's the
exact preprocessing of the original `meg-xl-med.ckpt` checkpoint (0.1–40 Hz
@ 50 Hz), which your own TFM §4.2 already flags as not fully Nyquist-valid
(effective band ends up ~0.1–25 Hz after antialiasing). Keeping it lets you
directly compare "what the base checkpoint actually saw" against every
Nyquist-compliant combo.

| fs | Nyquist (fs/2) | theta | alpha | beta | gamma | broadband_megxl |
|---|---|---|---|---|---|---|
| 50  | 25  | valid | valid | invalid | invalid | **anchor (invalid, kept intentionally)** |
| 100 | 50  | valid | valid | valid | valid | valid |
| 150 | 75  | valid | valid | valid | valid | valid |
| 200 | 100 | valid | valid | valid | valid | valid |
| 250 | 125 | valid | valid | valid | valid | valid |

Total: 22 Nyquist-respecting combos + 1 anchor = 23.

## 4. Seeds

5 per combo (placeholders below — swap for whatever you actually use):
`13, 42, 123, 2024, 31415`

## 5. Staged execution (recommended given prior VRAM headroom issues —
batch 6 previously hit ~98% VRAM and failed near step 5000)

- **Stage 1** — 1 seed × all 23 combos. Purpose: map out where performance
  actually degrades relative to the theoretical Nyquist ceiling — this *is*
  the empirical test of the "fs/5 usable bandwidth" heuristic your
  neuroengineer contact mentioned.
- **Stage 2** — remaining 4 seeds, but only on the combos you promote from
  Stage 1 (best performers, and/or combos near the suspected practical
  ceiling where you specifically want statistical confirmation). Use
  `run_eeg_chained_sweep.py`'s existing promotion-record mechanism for this
  rather than a flat re-run of everything.

## 6. Tokenizers

- `biocodec` — confirmed functional (default, wired in `factory.py`).
- `brainomni_base`, `brainomni_tiny`, `braintokenizer` — configs/checkpoints
  present but **not wired for encoding** (per `README_EEGXL.md`'s own
  Limitations section). Needs `factory.py` read + edit — blocked until the
  GitHub connector is reconnected.

## 7. Other hyperparameters (candidates, not yet confirmed against actual
config keys)

- Masking ratio / number of masked blocks (MEG-XL paper: 20 blocks, ~40%,
  but the Hydra key name in this repo's config isn't confirmed yet)
- Window length (150s default — shorter windows trade context for more
  samples)
- Learning rate / schedule
- RVQ codebook size (BioCodec `eeg_config.yaml` — key names not yet seen)
- Batch size / gradient accumulation (relevant given the VRAM history)

## 8. Open items pending repo reconnection

1. Confirm the Hydra key for random seed (assumed `seed=` in the launcher
   below — verify against `configs/train_criss_cross_eeg_multi_feq.yaml`).
2. Read `brainstorm/neuro_tokenizers/factory.py` and wire in at least one
   of BrainOmni base/tiny.
3. Read `brainstorm/neuro_tokenizers/biocodec/configs/eeg_config.yaml` to
   get real key names for the RVQ/masking hyperparameters in §7.
4. Cross-check this design against the actual `configs/eeg_sweep.yaml` /
   `configs/eeg_chained_sweep.yaml` / `scripts/make_eeg_sweep_plan.py` so
   the new launcher integrates with (rather than duplicates) the existing
   sweep infrastructure.
