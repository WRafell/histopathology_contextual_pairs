import torch

from src.lars import compute_learning_rate
from src.ssl_losses import combine_pair_losses, compute_barlow_twins_loss, compute_off_diagonal, compute_vicreg_loss


def test_off_diagonal() -> None:
    assert compute_off_diagonal(torch.arange(9.0).view(3, 3)).tolist() == [1, 2, 3, 5, 6, 7]


def test_identical_decorrelated_views_have_low_loss() -> None:
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(4096, 8, generator=generator)
    vicreg = compute_vicreg_loss(
        first=features, second=features, invariance_weight=25.0, variance_weight=25.0, covariance_weight=1.0
    )
    barlow = compute_barlow_twins_loss(first=features, second=features, off_diagonal_weight=0.005)
    shuffled = features[torch.randperm(4096, generator=generator)]
    barlow_shuffled = compute_barlow_twins_loss(first=features, second=shuffled, off_diagonal_weight=0.005)
    assert vicreg < 0.1
    assert barlow < 0.1 < barlow_shuffled


def test_combine_pair_losses() -> None:
    loss = combine_pair_losses(augmentation_loss=torch.tensor(2.0), context_loss=torch.tensor(4.0), context_weight=0.25)
    assert float(loss) == 2.5


def test_learning_rate_schedule() -> None:
    kwargs = {"total_steps": 100, "warmup_steps": 10, "peak_learning_rate": 1.0}
    assert compute_learning_rate(step=0, **kwargs) == 0.0
    assert compute_learning_rate(step=10, **kwargs) == 1.0
    assert compute_learning_rate(step=99, **kwargs) < 0.01
