import torch
from typing import List


def image_to_tensor(raw: List[List[float]]) -> torch.Tensor:
    x = torch.tensor(raw,
                     # normalizing weights
                     dtype=torch.float32).unsqueeze(0) / 255.0
    return x
