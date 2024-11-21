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
    proximity_patch: bool = args.positive_pair_sampling == 'closest_patch'
    dataset = args.dataset.lower()

    if preselected_samples:
        return f"ds={dataset}_pps={args.positive_pair_sampling}-level={args.proximity_level}_ssl={args.ssl_method}_encoder={args.backbone_name}"
    elif proximity_patch:
        return f"ds={dataset}_pps={args.positive_pair_sampling}-{args.minimum_distance}_ssl={args.ssl_method}_encoder={args.backbone_name}"
    else:
        return f"ds={dataset}_pps={args.positive_pair_sampling}_ssl={args.ssl_method}_encoder={args.backbone_name}"
