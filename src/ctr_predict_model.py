import torch
import torch.nn as nn
import torch.nn.functional as F


class CtrPredictionModule(torch.nn.Module):
    def __init__(self):
        super(CtrPredictionModule, self).__init__()
        self.content_dense1 = nn.Linear(768, 1)

    def forward(self, curr_pooled_outputs):
        """
        Input: 
            @curr_pooled_outputs: [Batch_size, 5, 768]
            @category_embedding: [Batch_size, 5, 300]
        Output:
            @ctr: [Batch_size, 5, 1]
        """
        content = F.gelu(self.content_dense1(curr_pooled_outputs))
        content = F.sigmoid(content)
        return content.squeeze(-1)

