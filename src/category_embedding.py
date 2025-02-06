import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.vocab as vocab

CATEGORY_CNT = 18
# category_list = ['lifestyle', 'health', 'news', 'sports', 'weather', 'entertainment', 'autos', 'travel', 'foodanddrink', 'tv', 'finance', 'movies', 'video', 'music', 'kids', 'middleeast', 'northamerica', 'games']
# cache = "./glove"
# glove = vocab.GloVe(name="42B", dim=300, cache=cache)
# category_ebd = []
# for category in category_list:
#     # if category:
#     #     vec1 = glove.vectors[glove.stoi["food"]]
#     #     vec2 = glove.vectors[glove.stoi["drink"]]
#     #     vec = (vec1 + vec2) / 2
#     # else:
#     vec = glove.vectors[glove.stoi[category]]
#     category_ebd.append(vec.numpy())
# category_ebd = np.array(category_ebd)
# print(category_ebd.shape)
# np.save("category_embedding/42B_category_embedding", category_ebd)


class PretrainedCategoryEmbedding():
    def __init__(self) -> None:
        super(PretrainedCategoryEmbedding, self).__init__()
        category_ebd = np.load("./category_embedding/840B_category_embedding.npy", allow_pickle=True)
        padding_ebd = np.mean(category_ebd, axis=0).reshape((1, -1))
        category_ebd = np.vstack((category_ebd, padding_ebd))
        # 
        category_ebd = torch.from_numpy(category_ebd)
        self.embedding = nn.Embedding(CATEGORY_CNT+1, 300)
        self.embedding.weight.data.copy_(category_ebd)
        self.embedding.weight.requires_grad = False

    def forward(self):
        # [batch_size * 1] - > [batch_size * embedding_size]
        return self.embedding

class CategoryEmbedding():
    def __init__(self) -> None:
        super(CategoryEmbedding, self).__init__()
        self.embedding = nn.Embedding(CATEGORY_CNT+1, 300)
    
    def forward(self):
        return self.embedding

