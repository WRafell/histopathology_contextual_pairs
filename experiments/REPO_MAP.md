# Repository map

Rewritten 2026-10-21 as a functional core with thin entry points; Hydra removed. Smoke-tested
end to end on synthetic data only; no real experiment has been run with this code.

## Camelyon16 metadata (`src/build_camelyon16_metadata.py`)

Inputs (soda, `/nas2/lait/1000_Members/wrafell/CAMELYON16`): `patches_20x_224_png/patch_manifest_shard*.csv`
(level-0 offsets), `derived/patch_grid_inputs/runs/abmil_hibou_224_43c8/manifest.csv` (split, slide label),
`toposlide/inputs/wsi_dim.csv` (slide magnification), `masks/<slide>_mask.tif`. Per slide: footprint =
224 * magnification / 20 level-0 px; grid index = round(offset / footprint), rejected if off-grid; tumor
fraction = annotated area within the footprint on the mask level nearest downsample 16; `label` =
fraction >= 0.5. Output `<output-dir>/csv/camelyon16_data.csv`.

## SSL pretraining (`src/train_ssl.py`)

1. Load metadata (`src/data.py::load_patch_metadata`); keep `subset == "train"`.
2. Anchors: all training patches or `max_anchors` sampled without replacement
   (`select_anchor_rows`, seeded by `seed`).
3. Neighbors: `src/neighbors.py::compute_neighbor_table` lists, for every training patch, the
   same-slide patches at each Chebyshev offset in `[pairing.min_distance, pairing.max_distance]`
   (any training patch can be a neighbor, not only anchors). Rows are then restricted to anchors.
4. Each epoch, `select_context_rows` picks the contextual positive of every anchor:
   - `augmentation` (E0): the anchor itself, so the third view is another augmentation and all
     conditions encode three views per step.
   - `random_neighbor` (E1): a uniform existing neighbor; anchors without neighbors fall back to
     themselves (logged as `self_pair_fraction`).
5. `PositivePairDataset` returns `(T1(anchor), T2(anchor), T2(context))`. The loss is
   `(1 - alpha) L(z_anchor, z_aug) + alpha L(z_anchor, z_ctx)` (`ssl_models.py::compute_positive_pair_loss`);
   the anchor is encoded once. Losses are computed in float32 under fp16 autocast with a `GradScaler`;
   overflowed steps are skipped and counted (`skipped_fp16_steps`).
6. LARS with linear warm-up (`optimizer.warmup_epochs`) and cosine decay to 0.1% of
   `base_learning_rate * batch_size / 256`, set per step. BYOL updates its EMA target after each step.
7. Outputs: `config.yaml`, `git_commit.txt`, `metrics.jsonl` (per epoch), `checkpoint.pt`
   (full SSL model, for continuation via `initial_checkpoint`; optimizer state is not restored), `encoder.pt`.

## Evaluation

- `src/extract_embeddings.py`: per-slide `slides/<slide>.pt` with fp16 embeddings, grid coords, and
  patch labels (`-1` when unknown), plus `manifest.csv`. Uses the SSL run's backbone and image size,
  deterministic resize + ImageNet normalization.
- `src/train_mil.py`: ABMIL (Ilse et al. attention, logits + cross-entropy) or mean pooling + linear,
  Adam with cosine decay, one slide per step, best epoch by validation loss; test accuracy,
  balanced accuracy, AUROC.
- `src/linear_probe.py`: linear classifier on labeled patch embeddings (labels `-1` and
  `ignored_labels` dropped), best epoch by validation loss, same test metrics.

## Randomness

| Source | Control |
| --- | --- |
| Anchor subsample | `np.random.default_rng(seed)` in `select_anchor_rows` |
| Neighbor choice per epoch | one `np.random.default_rng(seed)` owned by `train_ssl.main` |
| Batch order | `torch.Generator().manual_seed(seed + epoch)` per epoch loader |
| Augmentations | DataLoader worker torch RNG, derived from the loader generator |
| Model init | global torch RNG seeded by `seed_global_generators(seed)` |
| MIL / probe order | `torch.Generator().manual_seed(seed)` |

cuDNN kernels are not forced to be deterministic.

## Not implemented yet

- E2/E3 similarity-based neighbor selection with a frozen warm-up encoder.
- Pair-agreement and boundary-similarity analyses.
- DINO, TCGA-NSCLC metadata, multi-GPU training (run one job per GPU instead).
