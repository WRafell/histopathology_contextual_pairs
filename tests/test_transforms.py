import numpy as np
import torch
from PIL import Image

from src.transforms import make_view_transforms, rotate_by_random_quarter_turns


def test_quarter_turns_cover_all_rotations_exactly() -> None:
    torch.manual_seed(0)
    image = torch.arange(2 * 3 * 3).reshape(2, 3, 3)
    expected = {k: torch.rot90(image, k=k, dims=(-2, -1)) for k in range(4)}
    seen = set()
    for _ in range(100):
        rotated = rotate_by_random_quarter_turns(image)
        matches = [k for k, target in expected.items() if torch.equal(rotated, target)]
        assert len(matches) == 1
        seen.add(matches[0])
    assert seen == {0, 1, 2, 3}


def test_view_transforms_return_normalized_square_tensor() -> None:
    torch.manual_seed(0)
    pixels = np.random.default_rng(0).integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    for transform in make_view_transforms(image_size=224):
        view = transform(image)
        assert view.shape == (3, 224, 224)
        assert view.dtype == torch.float32
