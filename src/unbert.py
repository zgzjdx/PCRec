# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (C) 2021. The Chinese University of Hong Kong. All rights reserved.
#
# Authors: Qi Zhangqi <Huawei Noah's Ark Lab>
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
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from .modeling_bert import BertEncoder, BertPreTrainedModel, BertEmbeddings
from .pooling import BertPooler, get_attn_mask, AttentionPooler
from .attention import MultiHeadAttention, AdditiveAttention, PolyAttention


BertLayerNorm = torch.nn.LayerNorm


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


class UNBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        device = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        inputs_embeds = self.word_embeddings(input_ids)
        # batch_size * seq_max_len * hidden_size
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 论文里说没有用position embedding
        # batch_size * seq_max_len * hidden_size
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UNBertModel(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.args = args
        self.embeddings = UNBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.hist_embeddings = UNBertEmbeddings(config)
        # self.hist_encoder = BertEncoder(config)
        self.encoder = BertModel.from_pretrained(args.pretrain)
        self.hist_encoder = BertModel.from_pretrained(args.pretrain)
        self.bertpooler = BertPooler(config)
        self.dropout = DropoutWrapper(dropout_p=0.1)
        self.init_weights()
        self.poly_attn = PolyAttention(in_embed_dim=768, num_context_codes=32,
                                       context_code_dim=200)
        # if args.news_mode == 'attention':
        #     # self.news_attn = AttentionPooler(manager=self.config)
        #     self.self_attn = MultiHeadAttention(self.config.hidden_size, 12)
        #     self.news_attn = AdditiveAttention(config.hidden_size, config.hidden_size)
        #
        # self.un_attn = MultiHeadAttention(self.config.hidden_size, 12)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

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

    def forward(
        self,
        curr_input_ids=None,
        curr_type_ids=None,
        curr_input_mask=None,
        hist_input_ids=None,
        hist_token_type=None,
        hist_input_mask=None,
        hist_mask=None,
        device=None,
    ):
        # todo 是否有time-sequential的BERT？
        curr_input_shape = curr_input_ids.size()
        batch_size = curr_input_shape[0]
        num_clicked = curr_input_shape[1]
        curr_input_ids = curr_input_ids.view(batch_size * num_clicked, -1)
        curr_type_ids = curr_type_ids.view(batch_size * num_clicked, -1)
        curr_input_mask = curr_input_mask.view(batch_size * num_clicked, -1)
        # batch_size * seq_max_len * hidden_size
        # last-layer hidden state, (all hidden states), (all attentions)
        curr_sequence_outputs = self.encoder(
            input_ids=curr_input_ids,
            token_type_ids=curr_type_ids,
            attention_mask=curr_input_mask,
        )[0]
        # batch_size * seq_max_len * hidden_size, standard bert sequence output
        # batch_size * max_seq_len, dropout + pooling
        curr_pooled_outputs = self.pooler(curr_sequence_outputs, curr_input_mask, news_mode='cls')
        curr_pooled_outputs = self.dropout(curr_pooled_outputs).view(batch_size, num_clicked, -1)
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
        # get user representation, batch_size * feature_dim
        # position-aware inter-news representation
        # following with NRMS multi-head self-attention
        hist_pooled_outputs = hist_pooled_outputs.view(batch_size, self.args.hist_max_len, -1)
        hist_mask = hist_mask > 0
        # multi-interesting modeling
        hist_pooled_outputs = self.poly_attn(embeddings=hist_pooled_outputs, attn_mask=hist_mask, bias=None)

        return curr_pooled_outputs, hist_pooled_outputs
