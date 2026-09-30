# Repository map

Inspected/refactored 2026-09-30. No experiments were run. Dataset images and metadata are expected under `local/`; exact layout remains configurable by updating `data_root` and the metadata mapping once supplied.

## Active pipeline

- SSL pretraining: `python -m src.train_ssl_model`; Hydra loads `src/config.yaml`. Active sampling is `baseline`, `closest_patch`, or `random_neighbor`.
- Spatial neighbors: `src/sample_neighbors.py` groups training patches by slide and buckets candidates by maximum absolute coordinate difference (Chebyshev distance). `src/utils_self_supervised.py::NeighborDataset` selects buckets from `minimum_distance` through `maximum_distance`; the native PyTorch loop computes `(1-alpha)*augmentation_loss + alpha*context_loss` in `_calculate_loss` in `src/train_ssl_model.py`.
- Feature extraction: `python -m src.generate_embeddings`; embeddings and manifest go under `local/embeddings` and `local/csv`.
- Slide evaluation: `python -m src.train_mil_model`; currently ABMIL only. TransMIL, CLAM-SB, and mean pooling are not implemented.
- Patch-level linear probe: `src/utils_supervised.py` has dataset/trainer classes, but no command-line entry point.
- t-SNE/NMI analysis: no entry point found.

## Data, models, and reproduction

Metadata defaults to `local/csv/<dataset>_data.csv`; `data_root` is a config setting. Dataset keys retained from the source are Camelyon16, SEEDP stomach, SEEDP colon, `tcga`, and the `tcga-nsclc` alias. The TCGA alias currently maps to the source's `tcga_data.csv`; split definitions, public/private status, and validation membership cannot be established without the actual metadata and study configs. Relative `img_path` values resolve under `data_root`.

`src/config.yaml` retains source defaults (seed 0, ResNet18, patch resize 64, Barlow Twins, alpha 0.5, 100 epochs). The active native PyTorch trainer supports Barlow Twins, BYOL, SimSiam, and VICReg on one device. DINO model/loss components remain, but DINO training is not integrated because the active data path does not produce the required multi-crop batches.

No MICCAI Table 1/2 reproduction command, seed list, or results were present. The original shell script chained pretraining, feature extraction, and ABMIL, but depended on missing config/data paths and did not provide reproducible run logging. The refactored config resolves those paths under `local/`; exact dataset locations are still pending.
