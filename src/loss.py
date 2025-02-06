from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as torch_f

from src.utils import pairwise_cosine_similarity

def cl_loss(similarity, mask, device):
    # n为batch_size
    # n = similarity.shape[0]
    # print('sim shape', similarity.size())
    mask_pos = mask.to(device)
    # print('mask pos', mask_pos)
    # 不同类为1的矩阵
    mask_neg = (torch.ones_like(mask) - mask).to(device)
    # print('mask neg', mask_neg)
    # # 对角线为0其余全为1的矩阵
    # mask_diagonal_0 = torch.ones(n, n) - torch.eye(n, n)
    # print('mask_diagonal_0', mask_diagonal_0.size())
    exp_similarity = torch.exp(similarity).to(device)
    pos = torch.sum(exp_similarity * mask_pos, 1).to(device)
    neg = torch.sum(exp_similarity * mask_neg, 1).to(device)
    loss = -(torch.mean(torch.log(pos / (pos + neg)))).to(device)
    return loss

def get_cl_loss(sim_and_mask, device):
    # hist_cos_sim, category_same_mask, user_cos_sim, interest_same_mask = sim_and_mask[0], sim_and_mask[1], sim_and_mask[2], sim_and_mask[3]
    user_cos_sim2, user_same_mask = sim_and_mask[0], sim_and_mask[1]
    # category_cl_loss = cl_loss(hist_cos_sim, category_same_mask, device)
    # user_cl_loss = cl_loss(user_cos_sim, interest_same_mask, device)
    user_cl_loss2 = cl_loss(user_cos_sim2, user_same_mask, device)
    # return category_cl_loss, user_cl_loss
    return user_cl_loss2

def get_batch_loss(judge_loss, category_cl_loss, user_cl_loss, category_weight, user_weight):
    # batch_loss = judge_loss + category_weight * category_cl_loss + 0 * user_cl_loss
    batch_loss = judge_loss + user_cl_loss
    return batch_loss

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

    def forward(self, poly_attn: Tensor, logits: Tensor, labels: Tensor):
        r"""
        Compute batch loss

        Args:
            poly_attn: tensor of shape ``(batch_size, num_context_codes, embed_dim)``.
            logits: tensor of shape ``(batch_size, npratio + 1)``.
            labels: a one-hot tensor of shape ``(batch_size, npratio + 1)``.

        Returns:
            Loss value
        """
        disagreement_loss = pairwise_cosine_similarity(poly_attn, poly_attn, zero_diagonal=True).mean()
        targets = labels.argmax(dim=1)
        # nn.CrossEntrpyLoss内部自带有softmax
        rank_loss = self._criterion(logits, targets)
        total_loss = disagreement_loss + rank_loss

        return rank_loss

class CLLoss(nn.Module):
    def __init__(self, criterion):
        super(CLLoss, self).__init__()
        self._criterion = criterion
        self.sigma1 = nn.Parameter(torch.rand(1))
        self.sigma2 = nn.Parameter(torch.rand(1))
        self.sigma3 = nn.Parameter(torch.rand(1))

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

    @staticmethod
    def cl_loss(similarity, mask):
        # n为batch_size
        # n = similarity.shape[0]
        # print('sim shape', similarity.size())
        mask_pos = mask
        # print('mask pos', mask_pos)
        # 不同类为1的矩阵
        mask_neg = (torch.ones_like(mask) - mask)
        # print('mask neg', mask_neg)
        # # 对角线为0其余全为1的矩阵
        # mask_diagonal_0 = torch.ones(n, n) - torch.eye(n, n)
        # print('mask_diagonal_0', mask_diagonal_0.size())
        exp_similarity = torch.exp(similarity)
        pos = torch.sum(exp_similarity * mask_pos, 1)
        neg = torch.sum(exp_similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))
        return loss

    def get_loss1(self, batch_loss, category_loss, user_loss, args, epoch):
        loss = batch_loss
        if args.cl_category:
            loss += args.category_weight * category_loss
        if args.cl_user:
            loss += args.user_weight * user_loss
        return loss

    def get_loss2(self, batch_loss, category_loss, user_loss):
        loss = 1 / (self.sigma1 ** 2) * batch_loss + 1 / (self.sigma2 ** 2) * category_loss + 1 / (self.sigma3 ** 2) * user_loss + 2 * torch.log(self.sigma1) + 2 * torch.log(self.sigma2) + 2 * torch.log(self.sigma3)
        return loss

    def get_loss3(self, batch_loss, category_loss, user_loss):
        loss = 0.5 / (self.sigma1 ** 2) * batch_loss + 0.5 / (self.sigma2 ** 2) * category_loss + 0.5 / (self.sigma3 ** 2) * user_loss + 2 * torch.log(1 + self.sigma1 ** 2) + 2 * torch.log(1 + self.sigma2 ** 2) + 2 * torch.log(1 + self.sigma3 ** 2)
        return loss

    def forward(self, batch_score, click_label, sims_masks, args, epoch):
        b_loss = self._criterion(batch_score, click_label)
        category_loss = None
        user_loss = None
        if args.cl_category:
            category_loss = self.cl_loss(sims_masks[0],sims_masks[1])
            sims_masks.pop(1)
        if args.cl_user:
            user_loss = self.cl_loss(sims_masks[0], sims_masks[1])
        loss = self.get_loss1(b_loss, category_loss, user_loss, args, epoch)
        return loss, category_loss, user_loss
