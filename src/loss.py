from torch import Tensor
import torch.nn as nn
import torch.nn.functional as torch_f

from src.utils import pairwise_cosine_similarity


class Loss(nn.Module):
    def __init__(self, criterion):
        super(Loss, self).__init__()
        self._criterion = criterion

    @staticmethod
    def compute_eval_loss(poly_attn: Tensor, logits: Tensor, labels: Tensor):
        """
        Compute loss for evaluation phase

        Args:
            poly_attn: tensor of shape ``(batch_size, num_context_codes, embed_dim)``.
            logits: tensor of shape ``(batch_size, 1)``.
            labels: a binary tensor of shape ``(batch_size, 1)``.

        Returns:
            Loss value
        """
        disagreement_loss = pairwise_cosine_similarity(poly_attn, poly_attn, zero_diagonal=True).mean()
        rank_loss = -(torch_f.logsigmoid(logits) * labels).sum()
        total_loss = disagreement_loss + rank_loss

        return total_loss.item()

    def forward(self, dis_loss: Tensor, logits: Tensor, labels: Tensor):
        r"""
        Compute batch loss

        Args:
            poly_attn: tensor of shape ``(batch_size, num_context_codes, embed_dim)``.
            logits: tensor of shape ``(batch_size, npratio + 1)``.
            labels: a one-hot tensor of shape ``(batch_size, npratio + 1)``.

        Returns:
            Loss value
        """
        # disagreement_loss = pairwise_cosine_similarity(poly_attn, poly_attn, zero_diagonal=True).mean()
        targets = labels.argmax(dim=1)
        # nn.CrossEntrpyLoss内部自带有softmax
        rank_loss = self._criterion(logits, targets)
        # dis_loss = None
        if dis_loss is not None:
            rank_loss = dis_loss + rank_loss
        return rank_loss
