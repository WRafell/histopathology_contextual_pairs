# Open questions

- How was the validation split in `abmil_hibou_224_43c8/manifest.csv` drawn (seed, stratified)? It is
  slide-level; Camelyon16 has no patient IDs.
- ViT-S/16 uses the same LARS recipe as ResNet-18. Should ViT-S use AdamW instead?
- Batch size 256 at 224 px (three views per step) replaces the earlier 1024 at 64 px. Confirm the
  per-GPU batch size and epoch budget for the warm-up and continuation stages.
- Which TCGA-NSCLC split did the MICCAI paper use? A new patient-level 70/10/20 split
  (`configs/splits/tcga_nsclc_split.csv`) is used until it is known.

# Decisions (2026-10-21)

- The research overview's Experiment 1 (E0–E3, VICReg, Camelyon16) replaces the SimTriplet/SSLP
  head-to-head as the immediate priority.
- Encoders start from ImageNet weights; the paper must say so.
- Camelyon16 uses the existing 20x 224 px patches on soda at native size (no 256 px set exists).
- MICCAI numbers are not reproduced; experiments start from this code.
- E0 uses three augmented views per step (context = anchor), so that compute is matched with E1–E3.
  Plain two-view E0 (`pairing.context_weight=0`) is supported for a later standard-baseline run.

# Decisions (2026-10-28)

- TCGA-NSCLC uses primary-tumor diagnostic (DX) slides only, a fixed stratified patient-level 70/10/20 split
  (seed 0), and every DX slide of a patient as its own MIL bag. Patches use the Camelyon16 patcher.
