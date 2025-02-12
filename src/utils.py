import torch
from torch import Tensor


def pairwise_cosine_similarity(x: Tensor, y: Tensor, zero_diagonal: bool = False) -> Tensor:
    r"""
    Calculates the pairwise cosine similarity matrix

    Args:
        x: tensor of shape ``(batch_size, M, d)``
        y: tensor of shape ``(batch_size, N, d)``
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero

    Returns:
        A tensor of shape ``(batch_size, M, N)``
    """
    x_norm = torch.linalg.norm(x, dim=2, keepdim=True)
    y_norm = torch.linalg.norm(y, dim=2, keepdim=True)
    distance = torch.matmul(torch.div(x, x_norm), torch.div(y, y_norm).permute(0, 2, 1))
    if zero_diagonal:
        assert x.shape[1] == y.shape[1]
        mask = torch.eye(x.shape[1]).repeat(x.shape[0], 1, 1).bool().to(distance.device)
        distance.masked_fill_(mask, 0)

    return distance