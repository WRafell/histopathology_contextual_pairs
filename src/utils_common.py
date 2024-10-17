from omegaconf import OmegaConf
import torch
import random
import numpy as np


def set_manual_seed(random_seed: int):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_basename(args: OmegaConf):

    preselected_samples: bool = args.positive_pair_sampling == 'preselected_positive_pairs'

    if preselected_samples:
        return f"pps={args.positive_pair_sampling}-level={args.proximity_level}_ssl={args.ssl_method}_encoder={args.backbone_name}"
    else:
        return f"pps={args.positive_pair_sampling}_ssl={args.ssl_method}_encoder={args.backbone_name}"
