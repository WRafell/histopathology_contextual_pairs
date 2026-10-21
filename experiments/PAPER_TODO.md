# Paper changes required by implementation decisions

- State that ResNet-18 and ViT-S/16 encoders are initialized from ImageNet weights
  (torchvision `ResNet18_Weights.DEFAULT`; timm `vit_small_patch16_224.augreg_in21k_ft_in1k`), not trained from scratch.
- State that patches are 224x224 px at 20x, fed to the encoder without resizing (d=1 neighbors are ~112 um apart).
- State that patch tumor labels (diagnostics only) mark patches with >= 50% annotated tumor area, and that the
  validation split (27 slides) is drawn from the official training slides at slide level (no patient IDs).
- State that training uses fp16 mixed precision.
