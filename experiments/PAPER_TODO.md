# Paper changes required by implementation decisions

- State that ResNet-18 and ViT-S/16 encoders are initialized from ImageNet weights
  (torchvision `ResNet18_Weights.DEFAULT`; timm `vit_small_patch16_224.augreg_in21k_ft_in1k`), not trained from scratch.
- State that patches are extracted at 256x256 px, 20x, and resized to 224x224 for the encoder.
- State that training uses fp16 mixed precision.
