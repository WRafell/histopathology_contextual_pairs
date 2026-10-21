# Spatial Positive Pairs for Histopathology SSL

Self-supervised pretraining of patch encoders where, besides two augmented views of an
anchor patch, a spatially neighboring patch from the same whole-slide image acts as an
additional positive:

    L = (1 - alpha) * L(anchor, augmented anchor) + alpha * L(anchor, neighbor)

Frozen patch embeddings are then evaluated with slide-level MIL (ABMIL, mean pooling) and a
patch-level linear probe. Supported objectives are VICReg, Barlow Twins, and BYOL (all
negative-free); supported backbones are ResNet-18 and ViT-S/16, both initialized from ImageNet
weights. Training runs in fp16 mixed precision on CUDA.

## Setup

```bash
conda create -n contextual-pairs python=3.11 pip
conda activate contextual-pairs
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"   # pytest, ruff, pyrefly
```

## Data

Each dataset has one metadata file `<data_root>/csv/<dataset>_data.csv` with one row per patch:

| Column | Meaning |
| --- | --- |
| `img_path` | Patch image, absolute or relative to `data_root` |
| `subset` | `train`, `val`, or `test` (split at patient level) |
| `slide` | Slide identifier |
| `slide_label` | Integer slide class, `0..C-1` |
| `label` | Integer patch class, `-1`/empty when unknown (diagnostics only, never used for SSL) |
| `grid_x`, `grid_y` | Integer patch-grid coordinates; spatially adjacent patches differ by 1 |

Camelyon16 uses 224x224 px patches at 20x (no resize). Its metadata is built from the
patch manifests, the slide split, and the tumor masks (a patch is tumor when at least half
of its area is annotated):

```bash
sbatch scripts/slurm/build_camelyon16_metadata.sh   # writes runs/metadata/camelyon16_20x_224/
```

Then pass `--set data.data_root=runs/metadata/camelyon16_20x_224` to SSL training.

## Workflow

Every entry point takes `--config <yaml>`, a new `--output-dir` (an existing one is never
overwritten), and any number of `--set key=value` overrides. The resolved configuration and
the git commit are written to the output directory.

```bash
# 1. SSL pretraining (E0: pairing.strategy=augmentation, E1: random_neighbor)
python -m src.train_ssl --config configs/ssl_vicreg_resnet18.yaml --output-dir runs/ssl/vicreg-r18-e1

# Continue from a warm-up run
python -m src.train_ssl --config configs/ssl_vicreg_resnet18.yaml --output-dir runs/ssl/vicreg-r18-e1-cont \
    --set initial_checkpoint=runs/ssl/vicreg-r18-warmup/checkpoint.pt

# 2. Per-slide frozen embeddings (backbone and image size are read from the SSL run)
python -m src.extract_embeddings --config configs/embeddings.yaml --output-dir runs/embeddings/vicreg-r18-e1 \
    --set ssl_run_dir=runs/ssl/vicreg-r18-e1

# 3. Slide-level MIL (abmil | mean_pooling)
python -m src.train_mil --config configs/mil.yaml --output-dir runs/mil/vicreg-r18-e1-abmil \
    --set embedding_dir=runs/embeddings/vicreg-r18-e1 --set aggregator=abmil

# 4. Patch-level linear probe
python -m src.linear_probe --config configs/linear_probe.yaml --output-dir runs/probe/vicreg-r18-e1 \
    --set embedding_dir=runs/embeddings/vicreg-r18-e1
```

## Code layout

| Module | Role |
| --- | --- |
| `src/config.py` | Typed frozen configurations, YAML loading, `--set` overrides |
| `src/data.py` | Metadata loading, coordinates, anchor subsampling |
| `src/neighbors.py` | Chebyshev neighbor table and per-epoch contextual positive selection |
| `src/transforms.py`, `src/datasets.py` | SSL views and image datasets |
| `src/backbones.py`, `src/ssl_models.py`, `src/ssl_losses.py` | Encoders, SSL heads, pure loss functions |
| `src/lars.py`, `src/ssl_training.py` | LARS, learning-rate schedule, fp16 training epoch |
| `src/embeddings.py`, `src/mil.py`, `src/metrics.py` | Embedding storage, MIL models, metrics |
| `src/train_ssl.py`, `src/extract_embeddings.py`, `src/train_mil.py`, `src/linear_probe.py` | Entry points |

See [experiments/REPO_MAP.md](experiments/REPO_MAP.md) for the control flow and randomness.

## Checks

```bash
ruff format src tests && ruff check src tests && pyrefly check && pytest
```
