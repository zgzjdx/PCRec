import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.layer_norm = LayerNorm(size)
        self.dropout_1 = nn.Dropout(dropout, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class LayerNorm(nn.Module):
    # ref: https://github.com/pytorch/pytorch/issues/1959
    #   :https://arxiv.org/pdf/1607.06450.pdf
    def __init__(self, hidden_size, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.alpha = nn.parameter(torch.ones(1,1,hidden_size)) # gain g
        self.beta = nn.parameter(torch.zeros(1,1,hidden_size)) # bias b
        self.eps = eps

    def forward(self, x):
        """
        Args:
            :param x: batch * len * input_size

        Returns:
            normalized x
        """
        mu = torch.mean(x, 2, keepdim=True).expand_as(x)
        sigma = torch.std(x, 2, keepdim=True).expand_as(x)
        return (x - mu) / (sigma + self.eps) * self.alpha.expand_as(x) + self.beta.expand_as(x)
