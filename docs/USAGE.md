# Using the patched sweep

## Files
- `run_eeg_multi_training_sweep.sh` → replaces `scripts/run_eeg_multi_training_sweep.sh`
  (fully backward compatible — old band names and single-seed usage still work)
- `frequency_experiments_additions.yaml` → paste inside the existing
  `data.frequency_experiments:` block in
  `configs/train_criss_cross_eeg_multi_feq.yaml` (informational/doc only —
  the launcher script above is the actual source of truth for band values)

## Confirmed facts this is built on (read directly from the live repo)
- Tokenizer switching needs **no code changes** — `factory.py` already has a
  complete `BrainOmniTokenizerAdapter` and `load_neuro_tokenizer()` accepts
  `biocodec`, `brainomni_base`, `brainomni_tiny`, `braintokenizer`. The
  README_EEGXL.md line about this being unwired is stale.
- Seed is a genuine top-level Hydra key: `seed: 42` in
  `train_criss_cross_eeg_multi_feq.yaml` (there's also a separate
  `training.sampler_seed: 42` for the `RecordingShuffleSampler` — the
  patched script only overrides `seed=`; override `training.sampler_seed=`
  too if you want the data-shuffling order to vary with the seed as well).
- Masking hyperparameters are real, overridable keys:
  `model.mask_duration` (3.0s) and `model.num_subsegments_to_mask` (20,
  matching MEG-XL's ~40%).
- `broadband_megxl_fs50` is not just an "anchor label" — its l_freq/h_freq/
  target_sfreq are literally this config's own top-level `data.l_freq: 0.1`
  / `data.h_freq: 40.0` / `data.target_sfreq: 50.0` defaults, i.e. it
  reproduces the file's baseline setting exactly.

## Stage 1 — map the practical Nyquist ceiling (1 seed, all 23 combos)

```bash
EEG_BANDS="theta_4_8_fs50 theta_4_8_fs100 theta_4_8_fs150 theta_4_8_fs200 theta_4_8_fs250 \
alpha_8_13_fs50 alpha_8_13_fs100 alpha_8_13_fs150 alpha_8_13_fs200 alpha_8_13_fs250 \
beta_13_30_fs100 beta_13_30_fs150 beta_13_30_fs200 beta_13_30_fs250 \
gamma_30_49_fs100 gamma_30_49_fs150 gamma_30_49_fs200 gamma_30_49_fs250 \
broadband_megxl_fs50 broadband_megxl_fs100 broadband_megxl_fs150 broadband_megxl_fs200 broadband_megxl_fs250" \
EEG_TOKENIZERS="biocodec" \
EEG_SEED=42 \
EEG_INIT_MODES="scratch" \
bash scripts/run_eeg_multi_training_sweep.sh
```

23 bands × 1 tokenizer × 1 init mode × 1 seed = **23 runs**. Add
`EEG_INIT_MODES="scratch pretrained"` to also cover MEG-XL initialization
(46 runs) if you want that comparison in stage 1 too.

## Stage 2 — seeds + tokenizers on promoted combos only

Once you've picked winners from stage 1 (e.g. the combos closest to the
practical ceiling, or simply the best performers), rerun just those with
the remaining seeds and, if you want, more tokenizers:

```bash
EEG_BANDS="beta_13_30_fs150 gamma_30_49_fs250" \
EEG_TOKENIZERS="biocodec brainomni_base" \
EEG_SEEDS="13 42 123 2024 31415" \
EEG_INIT_MODES="scratch" \
bash scripts/run_eeg_multi_training_sweep.sh
```

`EEG_SEEDS` (plural) is the new list-valued override; `EEG_SEED` (singular)
still works unchanged for single-seed runs.

## Quick smoke test before committing GPU time

```bash
EEG_BANDS="theta_4_8_fs100" EEG_TOKENIZERS="biocodec" EEG_SEED=42 \
EEG_MAX_STEPS=20 WANDB_MODE=offline \
bash scripts/run_eeg_multi_training_sweep.sh
```

## Still open
- `configs/eeg_sweep.yaml` (the word-classification **evaluation** sweep,
  separate from this pretraining sweep) still has `seeds: [42]` — extend
  that list too once you're ready to evaluate the promoted checkpoints with
  multiple seeds.
- `training.sampler_seed` — decide whether to tie it to the same seed loop.
