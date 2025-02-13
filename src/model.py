import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import BertModel, BertForMaskedLM, BertTokenizer, AdamW
from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertOnlyMLMHead
from transformers.configuration_bert import BertConfig
from .attention import DotProductAttention, TargetAwareAttention, PolyAttention, AdditiveAttention
from .pooling import BertPooler
from .category_embedding import PretrainedCategoryEmbedding
from .utils import pairwise_cosine_similarity
from .new_modeling_bert import BertModelForPrompt
import sys
import os
import json
sys.path.append('..')


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


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class UNBERT(nn.Module):
    def __init__(self, pretrained, device, args, answer):
        super(UNBERT, self).__init__()
        self._pretrained = pretrained
        self._news_mode = args.news_mode
        self._max_len = args.seq_max_len
        self._tem_max_len = args.tem_max_len
        self._config = BertConfig.from_pretrained(self._pretrained)
        # self._config.vocab_size = vocab_size
        self.encoder = BertModel.from_pretrained(self._pretrained)
        # self.hist_encoder = BertModel.from_pretrained(self._pretrained)
        self.template_tokenizer = BertTokenizer.from_pretrained(self._pretrained)
        self.template_embedding = BertEmbeddings(self._config)
        self.template_encoder = BertForMaskedLM.from_pretrained(self._pretrained)
        self.vocab_size = None
        self.template_encoder.resize_token_embeddings(self.vocab_size)
        # self.template_encoder = BertEncoder(self._config)
        # self.template_cls = BertOnlyMLMHead(self._config)
        # self.template_encoder = BertModelForPrompt.from_pretrained(self._pretrained)
        self.bertpooler = BertPooler(self._config)  # BertPooler就是最后的全连接层
        self.dropout = DropoutWrapper(dropout_p=0.1)
        self.device = device
        self.args = args
        self.score_type = args.score_type
        # self.target_aware_attn = TargetAwareAttention(768)
        # prompt-num
        self.user_interest_num = 32
        self.poly_attn = PolyAttention(in_embed_dim=768, num_context_codes=self.user_interest_num,
                                       context_code_dim=200)
        # 热度模块
        # self.Popularity_Score = PopularityModel()
        # self.popularity_weight1 = nn.Linear(768, 1)
        # self.popularity_weight2 = nn.Linear(32, 1)
        # 类别
        self.category_embedding = PretrainedCategoryEmbedding().forward()
        self.all_category_ids = torch.arange(19)
        self.all_category_embedding = self.category_embedding(self.all_category_ids)
        self.prompt_bert_answer = answer
        self.prompt_bert_answer_id = self.template_tokenizer.encode(self.prompt_bert_answer, add_special_tokens=False)
        self.user_place = None,
        self.mask_place = None,
        self.candidate_place = None
        # 对比学习
        self.cos_sim = Similarity(temp=args.temp)
        # 对比学习模块
        self.cl_category = args.cl_category
        self.cl_user = args.cl_user
        self.cl_news_category = args.cl_news_category
        self.cl_news_subcategory = args.cl_news_subcategory
        self.user_same = args.user_same
        self.news2category = nn.Linear(768, 300)
        self.batch_idx = 0

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

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """Makes broadcastable attention mask and causal mask so that future and maked tokens are ignored.

        Arguments:
            attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
            input_shape: tuple, shape of input_ids
            device: torch.Device, usually self.device

        Returns:
            torch.Tensor with dtype of attention_mask.dtype
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self._config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def get_category_sim(self, hist_category_ids, batch_size, device, num_categories=18, num_candidates=50, sim_bar=0.5):
        user_clicks = torch.zeros((batch_size, num_categories), dtype=torch.float32).to(device)
        # print('hist_category_ids', hist_category_ids)
        for i in range(batch_size):
            user_clicks[i] = torch.bincount(hist_category_ids[i], minlength=num_categories)+1
        # print('user_clicks', user_clicks)
        min_clicks = torch.min(user_clicks.unsqueeze(1), user_clicks.unsqueeze(0)).to(device)
        sum_of_min_clicks = torch.sum(min_clicks, dim=2).to(device)
        overlap_matrix = sum_of_min_clicks / num_candidates
        overlap_matrix = overlap_matrix > sim_bar
        overlap_matrix = overlap_matrix + 0
        return overlap_matrix

    def forward(self, curr_input_ids, curr_type_ids, curr_input_mask, curr_category_ids, curr_subcategory_ids, hist_input_ids,
                hist_token_type, hist_input_mask, hist_mask, hist_category_ids, template_ids, template_token_type, template_mask,
                click_label):
        curr_category_embedding = self.category_embedding(curr_category_ids)
        hist_category_embedding = self.category_embedding(hist_category_ids)
        category_bias = pairwise_cosine_similarity(hist_category_embedding, curr_category_embedding)
        # sequence_output, sentence_output, pooled_output, (hidden_states), (attentions)
        curr_input_shape = curr_input_ids.size()
        # print("curr_input_shape", curr_input_shape)
        # curr_input_shape = batch_size * 5(一个正样本四个负样本) * seq_max_len(新闻的长度)
        batch_size = curr_input_shape[0]
        # print("batch_size", batch_size)
        num_clicked = curr_input_shape[1]
        # print(num_clicked)# 5

        curr_input_ids = curr_input_ids.view(batch_size * num_clicked, -1)
        # print("curr_input_ids.size", curr_input_ids.size())
        curr_type_ids = curr_type_ids.view(batch_size * num_clicked, -1)
        # print("curr_type_ids", curr_type_ids)
        curr_input_mask = curr_input_mask.view(batch_size * num_clicked, -1)
        # curr_sequence_outputs = self.template_embedding(
        #     input_ids=curr_input_ids,
        #     token_type_ids=curr_type_ids,
        #     # attention_mask=curr_input_mask,
        # )
        # print("curr_sequence_outputs", curr_sequence_outputs.size())
        # print(curr_sequence_outputs)
        curr_sequence_outputs = self.encoder(
            input_ids=curr_input_ids,
            token_type_ids=curr_type_ids,
            attention_mask=curr_input_mask,
        )[0]

        # curr_sequence_outputs = [5(1+4), 20(seq_max_len), 768(hidden_size)]
        # batch_size * seq_max_len * hidden_size, standard bert sequence output
        # batch_size * max_seq_len, dropout + pooling
        curr_pooled_outputs = self.pooler(curr_sequence_outputs, curr_input_mask, news_mode='cls')  # 这一部分是news_encoder所以直接输出

        # hist_input_ids.shape = [1, 10(hist_len), 20(seq_len)]
        hist_batch_size, num_candidates = hist_input_ids.size(0), hist_input_ids.size(1)
        # print(num_candidates) 50
        hist_input_ids = hist_input_ids.view(hist_batch_size * num_candidates, -1)
        hist_token_type = hist_token_type.view(hist_batch_size * num_candidates, -1)
        hist_input_mask = hist_input_mask.view(hist_batch_size * num_candidates, -1)
        # hist和curr用的news_encoder为啥不一样捏
        hist_encoder_outputs = self.encoder(
            input_ids=hist_input_ids,
            token_type_ids=hist_token_type,
            attention_mask=hist_input_mask
        )[0]
        # print("hist_encoder_outputs", hist_encoder_outputs.size())
        # print(hist_encoder_outputs)
        # hist_encoder_outputs.shape = [10(hist_len), 20(seq_len), 768(hidden_size)]
        hist_pooled_outputs = self.pooler(hist_encoder_outputs, hist_input_mask, news_mode='cls')
        hist_pooled_outputs_dropout = self.dropout(hist_pooled_outputs)

        curr_pooled_outputs = self.dropout(curr_pooled_outputs).view(batch_size, num_clicked, -1)  #
        hist_pooled_outputs = hist_pooled_outputs.view(hist_batch_size, num_candidates, -1)
        hist_without_attn = hist_pooled_outputs
        # get user representation, batch_size * feature_dim
        # position-aware inter-news representation
        # following with NRMS multi-head self-attention
        hist_mask = hist_mask > 0

        # multi-interesting modeling
        # ------------------CL-user-start---------------------
        sims_masks = []
        hist_pooled_outputs, interest_score, weights = self.poly_attn(embeddings=hist_pooled_outputs, attn_mask=hist_mask, bias=category_bias)
        # with open(os.path.join("visualize-data/news_weights.tsv"), 'w',
        #           encoding='utf-8') as file:
        #     weights = interest_score.view(batch_size * num_candidates, -1).cpu().detach().numpy()
        #     print(weights)
        #     print(weights.shape)
        #     cats = hist_category_ids.view(batch_size * num_candidates, -1).numpy()
        #     for i in range(len(weights)):
        #         emb = weights[i].tolist()
        #         cat = cats[i]
        #         file.write(f'{cat}\t{emb}\n')
        if self.cl_category or self.cl_user:
            user_cos_sim = self.cos_sim(hist_pooled_outputs.view(batch_size, -1).unsqueeze(1),
                                        hist_pooled_outputs.view(batch_size, -1).unsqueeze(0))
            sims_masks.append(user_cos_sim)
        if self.cl_category:
            hist_category_sim_mask = self.get_category_sim(hist_category_ids, hist_batch_size, self.device, 18, num_candidates, sim_bar = 0.75)
            sims_masks.append(hist_category_sim_mask)

        softmax_interest_score = F.softmax(interest_score, dim=2)
        sum_interest_score = torch.sum(softmax_interest_score, dim=1)
        selected_weights, selected_interests = torch.topk(sum_interest_score, k=self.user_same, dim=1)
        sorted_interests = selected_interests.sort(dim=1).values

        if self.cl_user:
            interest_same_mask = torch.all(sorted_interests.unsqueeze(1) == sorted_interests.unsqueeze(0), dim=2)
            interest_same_mask = interest_same_mask.int()
            # interest_same_mask = torch.eq(selected_interests.permute(1, 0), selected_interests)  # 32 32
            # interest_same_mask = interest_same_mask + 0
            sims_masks.append(interest_same_mask)

        # with open(os.path.join("visualize-data/users-history-batch-" + str(self.batch_idx) + ".tsv"), 'w',
        #           encoding='utf-8') as file:
        #     users_emb = hist_pooled_outputs.view(batch_size, -1)
        #     # top1interests = sorted_interests.view(batch_size, -1)
        #     users_emb = users_emb.cpu().detach().numpy()
        #     # top1interests = top1interests.cpu().numpy()
        #     saved_data = ''
        #     num_categories = 18
        #     user_clicks = torch.zeros((batch_size, num_categories), dtype=torch.float32).to(self.device)
        #     # print('hist_category_ids', hist_category_ids)
        #     for i in range(batch_size):
        #         user_clicks[i] = torch.bincount(hist_category_ids[i], minlength=num_categories)
        #     # print('user_clicks', user_clicks)
        #     for i in range(len(users_emb)):
        #         # top1 = top1interests[i][0]
        #         user_emb = users_emb[i].tolist()
        #         history = user_clicks[i].tolist()
        #         history = [int(h) for h in history]
        #         print(hist_category_ids[i])
        #         print(history)
        #         saved_data += f'{history}\t{user_emb}\n'
        #     file.write(saved_data)

        if self.cl_news_category:
            curr_news2category = self.news2category(curr_pooled_outputs)
            curr_news_category_sim = self.cos_sim(curr_news2category.view(batch_size * num_clicked, -1).unsqueeze(1), self.all_category_embedding.unsqueeze(0).to(self.device))
            curr_news_category_mask = torch.eq(curr_category_ids.view(batch_size * num_clicked, -1), self.all_category_ids.view(1, -1).to(self.device))
            curr_news_category_mask = curr_news_category_mask + 0

            hist_news2category = self.news2category(hist_without_attn)
            hist_news_category_sim = self.cos_sim(hist_news2category.view(hist_batch_size * num_candidates, -1).unsqueeze(1), self.all_category_embedding.unsqueeze(0).to(self.device))
            hist_news_category_mask = torch.eq(hist_category_ids.view(hist_batch_size * num_candidates, -1), self.all_category_ids.view(1, -1).to(self.device))
            hist_news_category_mask = hist_news_category_mask + 0

            news_category_sim = torch.concat([curr_news_category_sim, hist_news_category_sim], dim=0)
            news_category_mask = torch.concat([curr_news_category_mask, hist_news_category_mask], dim=0)
            sims_masks.append(news_category_sim)
            sims_masks.append(news_category_mask)

            # with open(os.path.join("visualize-data/currnews-batch-" + str(self.batch_idx) + ".tsv"), 'w',
            #           encoding='utf-8') as file:
            #     curr_news_emb = curr_news2category.view(batch_size * num_clicked, -1).cpu().detach().numpy()
            #     print(curr_news_emb)
            #     print(curr_news_emb.shape)
            #     cats = curr_category_ids.view(batch_size * num_clicked, -1).numpy()
            #     subcats = curr_subcategory_ids.view(batch_size * num_clicked, -1).numpy()
            #     for i in range(len(curr_news_emb)):
            #         emb = curr_news_emb[i].tolist()
            #         cat = cats[i]
            #         subcat = subcats[i]
            #         file.write(f'{cat}\t{subcat}\t{emb}\n')

            # if self.batch_idx == 0:
            #     with open(os.path.join("visualize-data/cat-emb-batch-" + str(self.batch_idx) + ".tsv"), 'w',
            #               encoding='utf-8') as file:
            #         for i in range(len(self.all_category_embedding)):
            #             # 使用.item()提取tensor中的数值
            #             vector = self.all_category_embedding[i].numpy().tolist()
            #             file.write(f'{i}\t{vector}\n')

        if self.cl_news_subcategory:
            curr_news_sim = self.cos_sim(curr_pooled_outputs.view(batch_size * num_clicked, -1).unsqueeze(1),
                                         curr_pooled_outputs.view(batch_size * num_clicked, -1).unsqueeze(0))
            curr_news_mask = torch.eq(curr_subcategory_ids.view(batch_size * num_clicked, -1).unsqueeze(1), curr_subcategory_ids.view(batch_size * num_clicked, -1).unsqueeze(0)).squeeze(2)
            curr_news_mask = curr_news_mask + 0
            sims_masks.append(curr_news_sim)
            sims_masks.append(curr_news_mask)

        self.batch_idx += 1
        # print(self.batch_idx)
        # ------------------CL-user-end---------------------

        # 修改了BertModelForPrompt，得到prediction_scores,大小(batch_size * num_clicked, sequence_length, config.vocab_size)
        embedding_output = self.template_embedding(input_ids=template_ids,
                                                   token_type_ids=template_token_type)
        '''
        for idx in range(batch_size):
            for idy in range(32):
                embedding_output[idx][self.user_place+idy] = hist_pooled_outputs[idx][idy]
        '''
        # prompt-num
        embedding_output[:, self.user_place: self.user_place + self.user_interest_num] = hist_pooled_outputs
        embedding_output = embedding_output.unsqueeze(dim=1)
        template_ids = template_ids.unsqueeze(dim=1)
        template_token_type = template_token_type.unsqueeze(dim=1)
        template_mask = template_mask.unsqueeze(dim=1)
        embedding_output_repeat = embedding_output.repeat(1, num_clicked, 1, 1)
        template_ids_repeat = template_ids.repeat(1, num_clicked, 1)
        template_token_type_repeat = template_token_type.repeat(1, num_clicked, 1)
        template_mask_repeat = template_mask.repeat(1, num_clicked, 1)
        '''
        for idx in range(batch_size):
            for idy in range(num_clicked):
                embedding_output_repeat[idx][idy][self.candidate_place] = curr_pooled_outputs[idx][idy]
        '''
        embedding_output_repeat[:, :num_clicked, self.candidate_place] = curr_pooled_outputs
        embedding_output_repeat = embedding_output_repeat.view(batch_size * num_clicked, -1, 768)
        template_ids_repeat = template_ids_repeat.view(batch_size * num_clicked, -1)
        template_token_type_repeat = template_token_type_repeat.view(batch_size * num_clicked, -1)
        template_mask_repeat = template_mask_repeat.view(batch_size * num_clicked, -1)
        prediction_scores = self.template_encoder(inputs_embeds=embedding_output_repeat,
                                        attention_mask=template_mask_repeat,
                                        token_type_ids=template_token_type_repeat)[0]
        # template_mask_repeat = template_mask.unsqueeze(dim=1).repeat(1, num_clicked, 1)
        # template_mask_repeat = template_mask_repeat.view(batch_size * num_clicked, -1)
        # template_extended_attention_mask = self.get_extended_attention_mask(template_mask_repeat,
        #                                                                     embedding_output_repeat.size(),
        #                                                                     self.device)
        # template_head_mask = self.get_head_mask(None, self._config.num_hidden_layers)
        # template_sequence_output = self.template_encoder(
        #     embedding_output_repeat,
        #     attention_mask=template_extended_attention_mask,
        #     head_mask=template_head_mask
        # )[0]
        # prediction_scores = self.template_cls(template_sequence_output)
        mask_logits = prediction_scores[:, self.mask_place, :]#.view(out_logits.size(0), -1, out_logits.size(-1))[:, -1, :]
        # print(self.vocab_size)
        # print(mask_logits.size())
        # print(self.prompt_bert_answer, self.prompt_bert_answer_id)
        answer_logits = mask_logits[:, self.prompt_bert_answer_id]
        # # for idx in range(batch_size):
        # #     for idy in range(32):
        # #         embedding_output[idx][self.user_place+idy] = hist_pooled_outputs[idx][idy]
        # out_logits = self.template_encoder( # template_sequence_outputs,
        #     input_ids=template_ids,
        #     token_type_ids=template_token_type,
        #     attention_mask=template_mask,
        #     hist_news=hist_pooled_outputs,
        #     curr_news=curr_pooled_outputs,
        #     batch_size=batch_size,
        #     num_clicked=num_clicked,
        #     user_place=self.user_place,
        #     mask_place=self.mask_place,
        #     candidate_place=self.candidate_place
        # )
        # #print(out_logits[:, mask_place, :].size())
        # mask_logits = out_logits[:, self.mask_place, :]#.view(out_logits.size(0), -1, out_logits.size(-1))[:, -1, :]
        #
        # # 将mask_place取出，得到mask_logits,大小(batch_size * num_clicked, config.vocab_size)
        # answer_logits = mask_logits[:, self.prompt_bert_answer_id]
        # 得到对应answer_id的值,大小(batch_size * num_clicked, 2)
        # 只需要计算分数，loss在外面计算
        matching_scores = answer_logits
        return hist_pooled_outputs, matching_scores, sims_masks

    @property
    def config(self):
        return self._config
