import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class PopularityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recency_dense1 = nn.Linear(1, 4)
        self.recency_dense2 = nn.Linear(4, 1)
        self.content_dense1 = nn.Linear(768, 32)
        self.content_dense2 = nn.Linear(32, 1)
        self.ctr_dense1 = nn.Linear(1, 1)
        self.category_dense1 = nn.Linear(300, 32)
        self.category_dense2 = nn.Linear(32, 1)
        self.weight = nn.Linear(4, 1)

    def forward(self, curr_pooled_outputs, recency, CTR, curr_category_embedding):
        recency = recency.float()
        recency = recency.unsqueeze(dim=2)
        recency_score = self.recency_dense1(1/(recency+1))
        recency_score = F.gelu(recency_score)
        recency_score = self.recency_dense2(recency_score)
        recency_score = F.gelu(recency_score)
        content_score = self.content_dense1(curr_pooled_outputs)
        content_score = F.gelu(content_score)
        content_score = self.content_dense2(content_score)
        content_score = F.gelu(content_score)
        CTR = CTR.unsqueeze(dim=2)
        ctr_socre = F.gelu(self.ctr_dense1(CTR))
        category_score = F.gelu(self.category_dense1(curr_category_embedding))
        category_score = F.gelu(self.category_dense2(category_score))
        popular_score = self.weight(torch.concat((ctr_socre, recency_score, content_score, category_score), dim=2))
        return popular_score
        