# Repository map

Rewritten 2026-10-21 as a functional core with thin entry points; Hydra removed. Smoke-tested
end to end on synthetic data only; no real experiment has been run with this code.

## Camelyon16 metadata (`src/build_camelyon16_metadata.py`)

Inputs (soda, `/nas2/lait/1000_Members/wrafell/CAMELYON16`): `patches_20x_224_png/patch_manifest_shard*.csv`
(level-0 offsets), `derived/patch_grid_inputs/runs/abmil_hibou_224_43c8/manifest.csv` (split, slide label),
`toposlide/inputs/wsi_dim.csv` (slide magnification), `masks/<slide>_mask.tif`. Per slide: footprint =
224 * magnification / 20 level-0 px; grid index = round(offset / footprint), rejected if off-grid; tumor
fraction = tumor-valued (2; masks are 0 background, 1 tissue, 2 tumor) area within the footprint on the mask level nearest downsample 16; `label` =
fraction >= 0.5. Output `<output-dir>/csv/camelyon16_data.csv`.

## TCGA-NSCLC split and metadata

- `src/make_tcga_nsclc_split.py` (`configs/tcga_nsclc_split.yaml`): reads the LUAD and LUSC GDC download
  manifests on soda, keeps diagnostic slides (slide type `DX*`) of primary tumors (sample type `01`), labels
  LUAD 0 and LUSC 1, and splits patients (barcode `TCGA-XX-XXXX`) per class into 70/10/20 train/val/test
  after one `np.random.default_rng(0)` shuffle per class in label order. Result committed as
  `configs/splits/tcga_nsclc_split.csv`: 1052 slides, 956 patients (train 334+334, val 48+48, test 96+96
  patients LUAD+LUSC; 730/98/224 slides).
- `scripts/link_tcga_nsclc_images.sh` links the split slides into `TCGA/NSCLC/images/`;
  `scripts/slurm/patch_tcga_nsclc.sh` runs the external patcher snapshot that made the Camelyon16 patches
  (20x, 224 px, stride 224, same tissue filter) and records its sha256 in the output.
- `src/build_tcga_nsclc_metadata.py` (`configs/tcga_nsclc_metadata.yaml`): base magnification = 10 / MPP
  (objective power as fallback), read from each WSI as the patcher does; footprint = 224 * base / 20;
  grid as for Camelyon16 (`src/patch_grid.py`). `label` is -1 (no patch annotations), so no linear probe on
  TCGA-NSCLC. Split slides without patches go to `slides_without_patches.csv`.

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
   - `pairing.context_weight: 0` drops the context view: items are `(T1(anchor), T2(anchor))` and the
     loss is plain two-view `L(z_anchor, z_aug)` (standard VICReg/Barlow/BYOL); no neighbor table is
     built and `self_pair_fraction` is logged as null.
5. `PositivePairDataset` returns `(T1(anchor), T2(anchor), T2(context))`. The loss is
   `(1 - alpha) L(z_anchor, z_aug) + alpha L(z_anchor, z_ctx)` (`ssl_models.py::compute_positive_pair_loss`);
   the anchor is encoded once. Views (`src/transforms.py`), in order: RandomResizedCrop to `image_size`
   (bicubic, area scale 0.8–1 for T1, 0.6–1 for T2), rotation by k·90° (k uniform in 0..3), horizontal
   and vertical flips (p 0.5 each), ColorJitter (brightness 0.4, contrast 0.4, saturation 0.2, hue 0.1;
   p 0.5), grayscale (p 0.1 T1, 0.2 T2), Gaussian blur (kernel 5×9, σ 0.1–5; p 0.2), ImageNet
   normalization. Elastic distortion is excluded on purpose (it changes nuclear morphology). Losses
   are computed in float32 under fp16 autocast with a `GradScaler`; overflowed steps are skipped and counted (`skipped_fp16_steps`).
6. LARS with linear warm-up (`optimizer.warmup_epochs`) and cosine decay to 0.1% of
   `base_learning_rate * batch_size / 256`, set per step. BYOL updates its EMA target after each step.
7. Multi-GPU: under `torchrun`, one process per GPU; `batch_size` is per GPU, the LR scales with the
   global batch (`batch_size * world_size`), a `DistributedSampler` splits the anchors, and gradients
   are averaged by DistributedDataParallel around `ssl_models.py::PositivePairLoss`. The SSL loss and
   BatchNorm statistics are per GPU (no embedding gather, no SyncBatchNorm). All processes draw the same
   contextual positives; only rank 0 writes outputs, and the logged loss is rank 0's.
8. Outputs: `config.yaml`, `git_commit.txt`, `metrics.jsonl` (per epoch), `checkpoint.pt`
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
| Batch order and per-GPU shard | `DistributedSampler(seed=seed)` with `set_epoch(epoch)` |
| Augmentations | DataLoader worker torch RNG, from a loader generator seeded by `SeedSequence([seed, epoch, rank])` |
| Model init | global torch RNG seeded by `seed_global_generators(seed)` |
| MIL / probe order | `torch.Generator().manual_seed(seed)` |

cuDNN kernels are not forced to be deterministic.

## Not implemented yet

- E2/E3 similarity-based neighbor selection with a frozen warm-up encoder.
- Pair-agreement and boundary-similarity analyses.
- DINO, multi-node training, global-batch SSL loss across GPUs.
