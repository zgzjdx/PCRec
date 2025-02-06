# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformers import BertModel
from .configuration_bert import BertConfig
from .attention import DotProductAttention, TargetAwareAttention, PolyAttention, SelfAdditiveAttention, MultiHeadAttention, MultiHeadMatrixAttention
from .pooling import BertPooler
from .popularity_model import PopularityModel
from .category_embedding import PretrainedCategoryEmbedding
from .utils import pairwise_cosine_similarity
from .ctr_predict_model import CtrPredictionModule
from .dynamic_intention import DynamicIntention
from .dynamic_intention_prob import DynamicIntentionProb


class DropoutWrapper(nn.Module):
    """
    This is a dropout wrapper which supports the fix mask dropout
    """
    def __init__(self, dropout_p=0, enable_vbp=True):
        super(DropoutWrapper, self).__init__()
        """variational dropout means fix dropout mask
        ref: https://discuss.pytorch.org/t/dropout-for-rnns/633/11
        """
        self.enable_variational_dropout = enable_vbp
        self.dropout_p = dropout_p

    def forward(self, x):
        """
            :param x: batch * len * input_size
        """
        if self.training == False or self.dropout_p == 0:
            return x

        if len(x.size()) == 3:
            mask = 1.0 / (1-self.dropout_p) * torch.bernoulli((1-self.dropout_p) * (x.data.new(x.size(0), x.size(2)).zero_() + 1))
            mask.requires_grad = False
            return mask.unsqueeze(1).expand_as(x) * x
        else:
            return F.dropout(x, p=self.dropout_p, training=self.training)


def compute_diversity_score(category_ids):
    '''
    category_ids: torch.tensor([bs, 50])
    '''
    score_list=[]
    for idx, item in enumerate(category_ids):
        count_list = torch.bincount(item)[:14]
        # 兴趣越多样, score越大
        s = torch.sum(count_list * (count_list-1)) / ((torch.sum(count_list) * (torch.sum(count_list)-1)) + 1e-6)
        score = 1 - s
        score_list.append(score)
    return torch.tensor(score_list)


class UNBERT(nn.Module):
    def __init__(self, pretrained, device, args):
        super(UNBERT, self).__init__()
        self._pretrained = pretrained
        self._news_mode = args.news_mode
        self._max_len = args.seq_max_len
        self._config = BertConfig.from_pretrained(self._pretrained)
        self.encoder = BertModel.from_pretrained(self._pretrained)
        self.hist_encoder = BertModel.from_pretrained(self._pretrained)
        self.k = 64
        self.bertpooler = BertPooler(self._config)  # BertPooler就是最后的全连接层
        self.dropout = DropoutWrapper(dropout_p=0.1)
        self.device = device
        self.args = args
        self.score_type = args.score_type
        self.target_aware_attn = TargetAwareAttention(768)
        self.poly_attn = PolyAttention(in_embed_dim=768, num_context_codes=32,
                                       context_code_dim=200)
        # self.multi_intention = MultiIntention(self.k*8, self.k, 768, device=device)
        self.dynamic_intention = DynamicIntentionProb(48, 4, 768, device=device)
        self.unique_intention = SelfAdditiveAttention()
        # 热度模块
        self.Popularity_Score = PopularityModel()
        self.popularity_weight1 = nn.Linear(768, 1)
        self.popularity_weight2 = nn.Linear(32, 1)
        # self.popularity_weight = nn.Linear(2, 1)
        # 类别
        self.category_embedding = PretrainedCategoryEmbedding().forward()
        #
        self.ctr_prediction = CtrPredictionModule()

    def pooler(self, input_sequence, attention_mask, news_mode):
        input_sequence = self.dropout(input_sequence)
        if news_mode == 'cls':
            token_embeddings = input_sequence
            return self.bertpooler(token_embeddings)
        elif news_mode == 'max':
            token_embeddings = input_sequence
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9
            max_over_time = torch.max(token_embeddings, 1)[0]
            return max_over_time
        elif news_mode == 'mean':
            token_embeddings = input_sequence
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / torch.sqrt(sum_mask)
        elif news_mode == 'attention':
            token_embeddings = input_sequence
            token_embeddings = self.self_attn(
                query=token_embeddings,
                key=token_embeddings,
                value=token_embeddings,
                attention_mask=attention_mask,
            )
            token_embeddings = self.news_attn(token_embeddings)
            return token_embeddings
        else:
            raise NotImplementedError

    # def self_gather_attention(self, hist_pooled_outputs, curr_pooled_outputs, hist_mask):
    #     '''
    #     hist_pooled_outputs: [bs, 50, 768]
    #     curr_pooled_outputs: [bs, 5, 768]
    #     hist_mask: [bs, 50]
    #     '''
    #     truncated=10
    #     hist_pooled_outputs, hist_mask=hist_pooled_outputs[:, :truncated, :], hist_mask[:, :truncated]
    #     temp_hist = hist_pooled_outputs.masked_fill_(~hist_mask.unsqueeze(dim=-1), 1e-30)
    #     query = torch.sum(temp_hist, dim=1, keepdim=True) # [bs, 1, 768]
    #     weight = torch.matmul(query, hist_pooled_outputs.permute(0, 2, 1))    # [bs, 1, 50]
    #     weight = weight.masked_fill_(~hist_mask.unsqueeze(dim=1), 1e-30)
    #     weight = torch.softmax(weight, dim=-1)  #[bs, 1, 50]
    #     gathered = torch.matmul(weight, hist_pooled_outputs)   # [bs, 1, 768]
    #     score = torch.matmul(gathered, curr_pooled_outputs.permute(0, 2, 1)).squeeze(dim=1)
    #     return score    #[bs, 1, 50)

    def forward(self, curr_input_ids, curr_type_ids, curr_input_mask, curr_category_ids, hist_input_ids,
                hist_token_type, hist_input_mask, hist_mask, hist_category_ids, CTR, recency, epoch):
        #  TODO: -loss(diversity_pred, diversity_score)
        curr_category_embedding = self.category_embedding(curr_category_ids)
        hist_category_embedding = self.category_embedding(hist_category_ids)
        diversity_score = compute_diversity_score(hist_category_ids).to(self.device)
        category_bias = pairwise_cosine_similarity(hist_category_embedding, curr_category_embedding)
        curr_input_shape = curr_input_ids.size()
        batch_size = curr_input_shape[0]
        num_clicked = curr_input_shape[1]   # 5
        curr_input_ids = curr_input_ids.view(batch_size * num_clicked, -1)
        curr_type_ids = curr_type_ids.view(batch_size * num_clicked, -1)
        curr_input_mask = curr_input_mask.view(batch_size * num_clicked, -1)
        curr_sequence_outputs = self.encoder(
            input_ids=curr_input_ids,
            token_type_ids=curr_type_ids,
            attention_mask=curr_input_mask,
        )[0]
        curr_pooled_outputs = self.pooler(curr_sequence_outputs, curr_input_mask, news_mode='cls')  # 这一部分是news_encoder所以直接输出
        curr_pooled_outputs = self.dropout(curr_pooled_outputs).view(batch_size, num_clicked, -1)   # 

        hist_batch_size, num_candidates = hist_input_ids.size(0), hist_input_ids.size(1)
        hist_input_ids = hist_input_ids.view(hist_batch_size * num_candidates, -1)
        hist_token_type = hist_token_type.view(hist_batch_size * num_candidates, -1)
        hist_input_mask = hist_input_mask.view(hist_batch_size * num_candidates, -1)
        hist_encoder_outputs = self.hist_encoder(
            input_ids=hist_input_ids,
            token_type_ids=hist_token_type,
            attention_mask=hist_input_mask
        )[0]
        hist_pooled_outputs = self.pooler(hist_encoder_outputs, hist_input_mask, news_mode='cls')
        hist_pooled_outputs = self.dropout(hist_pooled_outputs)
        hist_pooled_outputs = hist_pooled_outputs.view(hist_batch_size, num_candidates, -1)
        hist_mask = hist_mask > 0

        # unique_interest_scores=self.self_gather_attention(hist_pooled_outputs, curr_pooled_outputs, hist_mask)
        unique_interest_scores = self.unique_intention(hist_pooled_outputs, curr_pooled_outputs, hist_mask, len=30)
        poly_repr, selected_intention, selected_scores, dis_loss = self.dynamic_intention(hist_pooled_outputs, hist_mask, diversity_score, category_bias, epoch)
        # poly_repr = self.poly_attn(hist_pooled_outputs, hist_mask, bias=category_bias)
        # dis_loss = 0
        matching_scores = torch.matmul(curr_pooled_outputs, poly_repr.permute(0, 2, 1))
        # matching_scores = pairwise_cosine_similarity(curr_pooled_outputs, poly_repr)    # [b, 5, 768] [b, 4, 768]
        matching_scores = self.target_aware_attn(query=poly_repr, key=curr_pooled_outputs, value=matching_scores)       
        # matching_scores = torch.sigmoid(matching_scores)
        # print(matching_scores.shape)
        final_scores = 0.6 * unique_interest_scores + 0.4 * matching_scores
        # final_scores = 0.4 * matching_scores

        return dis_loss, final_scores
