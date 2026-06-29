# Estat actual del dataset i dels entrenaments

Hola Vicent i Alfons, he creat aquesta branca i he ordenat el desastre que tenia al github ajajaj. Aquest repositori conserva el README original de `MEG-XL` en `README_MEGXL.md`. Este fitxer resumeix l'estat actual dels datasets locals, els entrenaments EEG que s'han preparat i els resultats que hi ha generats fins ara.

## Dataset disponible ara mateix

En `datasets/` hi ha estes carpetes:

| Ruta | Estat observat | Ús principal en el codi |
|---|---:|---|
| `datasets/eegdash/data/nm000228` | BIDS parcial amb `sub-kent0034`, fitxers EEG i metadades | Lectura natural, tasques `delong` i `control` |
| `datasets/zuco2` | Directori present, sense fitxers visibles en l'escaneig actual | Lectura natural ZuCo, tasca `NR` |
| `datasets/sparrkulee` | Directori present, sense fitxers visibles en l'escaneig actual | Escolta, tasca `listeningActive` |
| `datasets/OpenNeuroEEG` | Directori present, sense fitxers visibles en l'escaneig actual | Contenidor genèric d'OpenNeuro EEG |
| `datasets/armeni` | Directori present, sense fitxers visibles en l'escaneig actual | Avaluacions MEG/word classification Armeni |
| `datasets/libribrain` | Cache local de HuggingFace amb metadades de Sherlock | Avaluacions MEG/word classification LibriBrain |

També hi ha `data/cache/zuco`. Les rutes exactes que espera la fase d'escolta del sweep són `datasets/OpenNeuroEEG_ds004408`, `datasets/OpenNeuroEEG_ds007808` i `datasets/sparrkulee`. En l'estat actual, les dos carpetes `OpenNeuroEEG_ds004408` i `OpenNeuroEEG_ds007808` no apareixen en `datasets/`.

Els pesos base que sí estan presents són:

- `checkpoints/baseline/meg-xl-med.ckpt` (de MEG-XL)
- `brainstorm/neuro_tokenizers/biocodec_ckpt.pt` (del tokenizer)

Els checkpoints intermedis que els fine-tuning usen per defecte no estan ara mateix en `checkpoints/`:

- `checkpoints/eeg_full_band_reading_then_listening_compare/.../eeg_full_band_0p1_50_fixed50_50hz_biocodec_from_scratch_listening_seed42/checkpoint_best.pt`
- `checkpoints/eeg_full_band_reading_then_listening_compare/.../eeg_full_band_0p1_50_fixed50_50hz_biocodec_pretrained_listening_seed42/checkpoint_best.pt`

Per tant, per replicar els fine-tuning des de zero cal primer executar el sweep de lectura -> escolta, o restaurar eixos checkpoints en les rutes esperades.

## Entrenament EEG full-band: lectura -> escolta

El llançador principal és `scripts/run_eeg_full_band_reading_then_listening_sweep.sh`. El script carrega la canalització modular de `scripts/eeg_full_band_pipeline/` i executa dos pipelines:

- `pretrained`: inicialitza la lectura amb `checkpoints/baseline/meg-xl-med.ckpt`.
- `from_scratch`: inicialitza la lectura amb pesos aleatoris.

Els dos pipelines fan primer lectura i només passen a escolta si la lectura acaba correctament. La configuració comuna és:

- banda `0.1-50 Hz` (MEG-XL és fins a 40 ja l'he canviat)
- freqüència objectiu `50 Hz`
- tokenitzador `BioCodec`
- llavor per defecte `42`
- dos GPUs per defecte amb `EEG_GPUS="0 1"`
- cache principal `data/cache/eeg_preprocessed`

La fase de lectura usa `configs/train_criss_cross_eeg_reading_continuous.yaml`:

- EEGDash `delong` i `control`
- ZuCo `NR`
- split per subjecte

La fase d'escolta usa `configs/train_criss_cross_eeg_listening_continuous.yaml`:

- OpenNeuro `ds004408`, tasca `listening`
- OpenNeuro `ds007808`, tasques `listening` i `listeningcovert`
- SparrKULee `listeningActive`

Cada pipeline guarda el millor checkpoint de lectura i el passa com a `promoted_checkpoint` a la fase d'escolta. Les eixides per defecte es creen en:

- `results/eeg_full_band_reading_then_listening_compare/<RUN_ID>`
- `logs/eeg_full_band_reading_then_listening_compare/<RUN_ID>`
- `checkpoints/eeg_full_band_reading_then_listening_compare/<RUN_ID>`

## Fine-tuning three-way en Alice

El script `scripts/run_alice_three_way_finetuning.sh` compara tres inicialitzacions de manera seqüencial en una GPU:

| Ordre | Etiqueta | Inicialització |
|---:|---|---|
| 1 | `random_init` | arquitectura CrissCross aleatòria |
| 2 | `eeg_from_scratch` | checkpoint EEG entrenat de zero en lectura -> escolta |
| 3 | `eeg_pretrained` | checkpoint EEG inicialitzat des de MEG-XL i entrenat en lectura -> escolta |

La configuració usa `datasets/alice_eeg` com a arrel, subjectes `main` de l'Alice EEG, split sense fuga per text/sentence, 50 èpoques, batch size 1 i selecció final del millor checkpoint de validació. Avalua Top-10 amb vocabularis de 50, 250 i 601 paraules; el vocabulari de 601 paraules permet comparar amb Chen et al.

Resultat consolidat disponible:

- `results/alice_three_way/20260627_123237`
- execució completada el 27 de juny de 2026
- `random_init`: 12:32:37 -> 14:34:38
- `eeg_from_scratch`: 14:34:38 -> 19:09:44
- `eeg_pretrained`: 19:09:44 -> 23:44:13

## Fine-tuning three-way en Weissbart

El script `scripts/run_weissbart_three_way_finetuning.sh` fa la mateixa comparacio three-way sobre Weissbart EEG:

| Ordre | Etiqueta | Inicialització |
|---:|---|---|
| 1 | `random_init` | arquitectura CrissCross aleatòria |
| 2 | `eeg_from_scratch` | checkpoint EEG entrenat de zero en lectura -> escolta |
| 3 | `eeg_pretrained` | checkpoint EEG inicialitzat des de MEG-XL i entrenat en lectura -> escolta |

La configuració usa `datasets/WeissbartEEG` com a arrel, split sense fuga per sentence, 50 èpoques, batch size 1 i avaluació Top-10 amb vocabularis de 50 i 250 paraules.

Resultat consolidat disponible:

- `results/weissbart_three_way/20260626_130140`
- execució completada el 26 de juny de 2026
- `random_init`: 13:01:40 -> 14:12:33
- `eeg_from_scratch`: 14:12:33 -> 16:25:55
- `eeg_pretrained`: 16:25:55 -> 17:55:58

## Resultats aconseguits fins ara

Les columnes de 50 i 250 paraules mostren `balanced_top10_accuracy`. Les columnes d'Alice amb 601 paraules mostren les exactituds Top-1/Top-10 del CSV de comparació amb Chen et al. Les comparacions estadístiques Welch no són informatives encara, perquè només hi ha una llavor per model (`n=1`).

### Alice EEG

| Model | Top-10, 50 paraules | Top-10, 250 paraules | Top-1, 601 paraules | Top-10, 601 paraules |
|---|---:|---:|---:|---:|
| `random_init` | 18.64% | 3.85% | 0.30% | 2.61% |
| `eeg_from_scratch` | 82.61% | 78.40% | 61.42% | 75.50% |
| `eeg_pretrained` | 88.72% | 81.15% | 71.21% | 77.81% |
| `preprint_arxiv` | — | — | 4.10% | 26.82% |


En Alice, el checkpoint EEG inicialitzat des de MEG-XL és el millor en totes les mètriques principals. En vocabulari de 601 paraules, supera clarament la referència de Chen et al. indicada en el codi: Top-1 `4.10%` i Top-10 `26.82%` en validació.

### Weissbart EEG

| Model | Top-10, 50 paraules | Top-10, 250 paraules |
|---|---:|---:|
| `random_init` | 19.57% | 4.49% |
| `eeg_from_scratch` | 43.37% | 17.13% |
| `eeg_pretrained` | 47.97% | 19.23% |

En Weissbart, el checkpoint EEG inicialitzat des de MEG-XL també queda per damunt del checkpoint EEG entrenat de zero i de la inicialització aleatòria.

## Com replicar-ho

1. Verifica els pesos base:

```bash
test -f checkpoints/baseline/meg-xl-med.ckpt
test -f brainstorm/neuro_tokenizers/biocodec_ckpt.pt
```

2. Prepara les rutes de dades que falten si vols repetir tota la cadena:

```bash
datasets/eegdash/data
datasets/zuco2
datasets/OpenNeuroEEG_ds004408
datasets/OpenNeuroEEG_ds007808
datasets/sparrkulee
datasets/alice_eeg
datasets/WeissbartEEG
```

3. Executa el sweep EEG lectura -> escolta. Per defecte usa dos GPUs:

```bash
EEG_GPUS="0 1" bash scripts/run_eeg_full_band_reading_then_listening_sweep.sh
```

Si ja tens el preprocessat i vols saltar-lo:

```bash
EEG_SKIP_PREPROCESS=true EEG_GPUS="0 1" bash scripts/run_eeg_full_band_reading_then_listening_sweep.sh
```

4. Localitza els dos `checkpoint_best.pt` finals de la fase d'escolta i, si no estan en les rutes per defecte dels scripts de fine-tuning, passa'ls com a variables d'entorn:

```bash
SCRATCH_EEG_CHECKPOINT=/ruta/al/checkpoint_from_scratch.pt \
PRETRAINED_EEG_CHECKPOINT=/ruta/al/checkpoint_pretrained.pt \
bash scripts/run_weissbart_three_way_finetuning.sh
```

5. Replica Alice amb els mateixos checkpoints:

```bash
SCRATCH_EEG_CHECKPOINT=/ruta/al/checkpoint_from_scratch.pt \
PRETRAINED_EEG_CHECKPOINT=/ruta/al/checkpoint_pretrained.pt \
bash scripts/run_alice_three_way_finetuning.sh
```

6. Revisa les eixides:

```bash
find results/weissbart_three_way -maxdepth 3 -type f | sort
find results/alice_three_way -maxdepth 3 -type f | sort
```

Els CSV principals són:

- `megxl_paper_metrics_summary.csv`
- `megxl_pairwise_welch_tests.csv`
- `weissbart_three_way_test_metrics.csv`
- `alice_three_way_test_metrics.csv`
- `alice_reference_three_way_comparison.csv`


## Que estic fent ara?

OpenNeuro ds007808 sí que té un parell de treballs publicats sobre el dataset en resultats en els que puc comparar quantitativament. De forma que l'he llevat de l'entrenament i ja l'he preparat en format word_alligned per a poder fer fine-tuning sobre eixe dataset i comparar per a tindre ja resultats finals.

En memoria/ estic redactant poquet a poquet però no ho tingueu en compte, tinc la memòria oficial a l'overleaf i Vicent me l'està corregint. Qualsevol cosa em dieu.