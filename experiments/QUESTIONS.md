# Open questions

- Where are the Camelyon16 patches and `camelyon16_data.csv` (local path or cluster alias + remote root)?
  How are the train/val/test splits defined, and is the validation split patient-disjoint?
- Camelyon16 `label` encoding: which values mean tumor, non-tumor, and unannotated?
- ViT-S/16 uses the same LARS recipe as ResNet-18. Should ViT-S use AdamW instead?
- Batch size 256 at 224 px (three views per step) replaces the earlier 1024 at 64 px. Confirm the
  per-GPU batch size and epoch budget for the warm-up and continuation stages.

# Decisions (2026-10-21)

- The research overview's Experiment 1 (E0–E3, VICReg, Camelyon16) replaces the SimTriplet/SSLP
  head-to-head as the immediate priority.
- Encoders start from ImageNet weights; the paper must say so.
- Patches are 256 px at 20x, resized to 224; the paper must say so.
- MICCAI numbers are not reproduced; experiments start from this code.
- E0 uses three augmented views per step (context = anchor), so that compute is matched with E1–E3.
