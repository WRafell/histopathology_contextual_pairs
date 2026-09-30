# Contextual Positive Pairs for Histopathology

Code for self-supervised representation learning with spatially contextual positive patches, plus patch embedding extraction and slide-level multiple instance learning (MIL).

The active training pipeline uses native PyTorch. Training currently supports Barlow Twins, BYOL, SimSiam, and VICReg with a single device. Positive-pair sampling options are `baseline`, `closest_patch`, and `random_neighbor`. DINO components remain in the source tree, but DINO training and evaluation are not integrated. MIL currently supports ABMIL.

## Setup

Use Python 3.11 with the pinned dependencies:

```bash
conda create -n contextual-pairs python=3.11 pip
conda activate contextual-pairs
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for your machine if the pinned Torch wheels do not match its CPU/CUDA setup.

## Data layout

The default data root is `local/`. Dataset metadata CSVs belong in `local/csv/`:

| Dataset key | Metadata file |
| --- | --- |
| `camelyon16` | `camelyon16_data.csv` |
| `tcga` / `tcga-nsclc` | `tcga_data.csv` |
| `seedp-stomach` | `seedp-stomach_data.csv` |
| `seedp-colon` | `seedp-colon_data.csv` |

The metadata must include the columns consumed by the scripts, including `img_path`, `subset`, `slide`, `slide_label`, and patch `label`. Camelyon16 patch coordinates are inferred from image filenames; other datasets need a `coords` column containing coordinate pairs. Relative image paths are resolved under the data root. Neighbor preparation writes `local/csv/sampled_<dataset>.csv`; model weights, embeddings, and evaluation outputs are also written under `local/`.

Dataset files are not included in this repository. Set `data_root=/path/to/data` to use another location. The data manifest schemas and split definitions should be confirmed for your dataset before training.

## Workflow

Run commands from the repository root. Hydra options can be appended as `key=value` overrides.

1. Prepare spatial neighbors from the dataset metadata:

   ```bash
   python -m src.sample_neighbors dataset=camelyon16 data_root=local
   ```

2. Pretrain an encoder. For example, `closest_patch` samples a spatial neighbor and uses the configured distance range:

   ```bash
   python -m src.train_ssl_model \
     dataset=camelyon16 data_root=local \
     ssl_method=barlow_twins positive_pair_sampling=closest_patch
   ```

   Set `positive_pair_sampling=baseline` for augmentation-only pairs or `random_neighbor` to sample a random available neighbor. The default config is `src/config.yaml`; it also defines the seed, backbone, optimizer, image resize, batch size, and training schedule.

3. Extract patch embeddings using the same dataset and model settings as pretraining:

   ```bash
   python -m src.generate_embeddings \
     dataset=camelyon16 data_root=local \
     ssl_method=barlow_twins positive_pair_sampling=closest_patch
   ```

4. Train and evaluate the configured ABMIL classifier on the frozen embeddings:

   ```bash
   python -m src.train_mil_model \
     dataset=camelyon16 data_root=local \
     ssl_method=barlow_twins positive_pair_sampling=closest_patch
   ```

Keep model-defining overrides consistent between these commands so the same checkpoint and embedding manifest are selected.

## Repository notes

- `src/` contains the training, sampling, embedding, and MIL code.
- `experiments/REPO_MAP.md` summarizes the implementation and known gaps; `experiments/QUESTIONS.md` tracks unresolved dataset and experiment-plan questions.
- `src/ORIGIN.md` records the imported source revision and current integration limits.
- Do not start experiments until the local data paths and the Phase 1 baseline implementations in the experiment plan are resolved.
