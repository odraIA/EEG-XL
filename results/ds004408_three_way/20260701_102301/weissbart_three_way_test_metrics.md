| model | model_display | run_dir | seed | train_pct | retrieval_size | top_k | balanced_top_k_accuracy | top_k_accuracy | chance_accuracy | balanced_accuracy_x_chance | accuracy_x_chance | n_samples | n_skipped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_init | Arquitectura aleatoria | logs/word_classification_ds004408_eeg/20260701_102301/random_init | 42 | 1.000000 | 50 | 10 | 0.200000 | 0.249289 | 0.200000 | 1.000000 | 1.246445 | 20045 | 15105 |
| random_init | Arquitectura aleatoria | logs/word_classification_ds004408_eeg/20260701_102301/random_init | 42 | 1.000000 | 250 | 10 | 0.040541 | 0.143243 | 0.040000 | 1.013514 | 3.581081 | 28120 | 7030 |
| eeg_from_scratch | Checkpoint EEG desde cero | logs/word_classification_ds004408_eeg/20260701_102301/eeg_from_scratch | 42 | 1.000000 | 50 | 10 | 0.612575 | 0.642754 | 0.200000 | 3.062875 | 3.213769 | 20045 | 15105 |
| eeg_from_scratch | Checkpoint EEG desde cero | logs/word_classification_ds004408_eeg/20260701_102301/eeg_from_scratch | 42 | 1.000000 | 250 | 10 | 0.209465 | 0.421693 | 0.040000 | 5.236616 | 10.542319 | 28120 | 7030 |
| eeg_pretrained | Checkpoint EEG preentrenado | logs/word_classification_ds004408_eeg/20260701_102301/eeg_pretrained | 42 | 1.000000 | 50 | 10 | 0.612068 | 0.625193 | 0.200000 | 3.060340 | 3.125966 | 20045 | 15105 |
| eeg_pretrained | Checkpoint EEG preentrenado | logs/word_classification_ds004408_eeg/20260701_102301/eeg_pretrained | 42 | 1.000000 | 250 | 10 | 0.206917 | 0.409175 | 0.040000 | 5.172936 | 10.229374 | 28120 | 7030 |
