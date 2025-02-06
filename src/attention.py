import math
import numpy as np
import torch
from torch import _softmax_backward_data
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """
    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, value):
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)
        score = torch.bmm(query, value.transpose(1, 2))
        attn = nn.functional.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        attn = self.dropout(attn)
        context = torch.bmm(attn, value)

        return context


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value, mask):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (:obj:`torch.tensor`): The input tensor that will apply softmax.
        mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
    """

    @staticmethod
    def forward(self, input, mask, dim):
        self.dim = dim
        rmask = ~(mask.bool())
        output = input.masked_fill(rmask, float("-inf"))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        (output,) = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
        return inputGrad, None, None


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, head_num):
        super().__init__()
        self.head_num = head_num
        assert hidden_dim % head_num == 0, "hidden_dim {} must divide head_num {}".format(hidden_dim, head_num)
        self.head_dim = hidden_dim // head_num
        self.hidden_dim = hidden_dim

        self.keyProject = nn.Linear(hidden_dim, self.head_dim * head_num)
        self.valueProject = nn.Linear(hidden_dim, self.head_dim * head_num)
        self.queryProject = nn.Linear(hidden_dim, self.head_dim * head_num)

        nn.init.xavier_normal_(self.keyProject.weight)
        nn.init.xavier_normal_(self.valueProject.weight)
        nn.init.xavier_normal_(self.queryProject.weight)

    def transpose_for_scores(self, x):
        """
        transpose the head_num dimension, to make every head operates in parallel
        """
        # [B, seq_len, head_num, -1]
        new_x_shape = x.size()[:-1] + (self.head_num, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, key, value, query, attention_mask=None):
        """ customized bert self attention, attending to the references

        Args:
            hidden_states: normally encoded candidate news, [batch_size, signal_length, hidden_dim]

        Returns:
            attn_output: [batch_size, signal_length, value_dim * num_head]
        """
        # [batch_size, head_num, *, head_dim]
        query = self.transpose_for_scores(self.keyProject(query))
        key = self.transpose_for_scores(self.keyProject(key))
        value = self.transpose_for_scores(self.valueProject(value))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query, key.transpose(-1, -2))

        # [bs, hn, sl+1, *]
        attention_scores = (attention_scores / math.sqrt(self.head_dim))
        if attention_mask is not None:
            attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        else:
            attention_probs = torch.softmax(attention_scores, -1)

        attn_output = torch.matmul(attention_probs, value)

        # [batch_size, signal_length, head_num, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        new_shape = attn_output.size()[:-2] + (self.head_dim * self.head_num,)
        attn_output = attn_output.view(*new_shape)

        return attn_output


# class AdditiveAttention(torch.nn.Module):
#     """
#     A general additive attention module.
#     Originally for NAML.
#     """
#     def __init__(self,
#                  query_vector_dim,
#                  candidate_vector_dim,
#                  writer=None,
#                  tag=None,
#                  names=None):
#         super(AdditiveAttention, self).__init__()
#         self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
#         self.attention_query_vector = nn.Parameter(
#             torch.empty(query_vector_dim).uniform_(-0.1, 0.1))
#         # For tensorboard
#         self.writer = writer
#         self.tag = tag
#         self.names = names
#         self.local_step = 1
#
#     def forward(self, candidate_vector):
#         """
#         Args:
#             candidate_vector: batch_size, candidate_size, candidate_vector_dim
#         Returns:
#             (shape) batch_size, candidate_vector_dim
#         """
#         # batch_size, candidate_size, query_vector_dim
#         temp = torch.tanh(self.linear(candidate_vector))
#         # batch_size, candidate_size
#         candidate_weights = F.softmax(torch.matmul(
#             temp, self.attention_query_vector),
#                                       dim=1)
#         if self.writer is not None:
#             assert candidate_weights.size(1) == len(self.names)
#             if self.local_step % 10 == 0:
#                 self.writer.add_scalars(
#                     self.tag, {
#                         x: y
#                         for x, y in zip(self.names,
#                                         candidate_weights.mean(dim=0))
#                     }, self.local_step)
#             self.local_step += 1
#         # batch_size, candidate_vector_dim
#         target = torch.bmm(candidate_weights.unsqueeze(dim=1),
#                            candidate_vector).squeeze(dim=1)
#         return target


class TargetAwareAttention(nn.Module):
    """Implementation of target-aware attention network"""
    def __init__(self, embed_dim: int):
        r"""
        Initialization

        Args:
            embed_dim: The number of features in query and key vectors
        """
        super().__init__()
        self.linear = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

    def forward(self, query, key, value):
        r"""
        Forward propagation

        Args:
            query: tensor of shape ``(batch_size, num_context_codes, embed_dim)``
            key: tensor of shape ``(batch_size, num_candidates, embed_dim)``
            value: tensor of shape ``(batch_size, num_candidates, num_context_codes)``

        Returns:
            tensor of shape ``(batch_size, num_candidates)``
        """
        proj = F.gelu(self.linear(query))
        weights = F.softmax(torch.matmul(key, proj.permute(0, 2, 1)), dim=2)
        outputs = torch.mul(weights, value).sum(dim=2)

        return outputs


class PolyAttention(nn.Module):
    r"""
    Implementation of Poly attention scheme that extracts `K` attention vectors through `K` additive attentions
    """
    def __init__(self, in_embed_dim: int, num_context_codes: int, context_code_dim: int):
        r"""
        Initialization

        Args:
            in_embed_dim: The number of expected features in the input ``embeddings``
            num_context_codes: The number of attention vectors ``K``
            context_code_dim: The number of features in a context code
        """
        super().__init__()
        self.linear = nn.Linear(in_features=in_embed_dim, out_features=context_code_dim, bias=False)
        self.context_codes = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_context_codes, context_code_dim),
                                                                  gain=nn.init.calculate_gain('tanh')))

    def forward(self, embeddings, attn_mask, bias=None):
        r"""
        Forward propagation

        Args:
            embeddings: tensor of shape ``(batch_size, his_length, embed_dim)``
            attn_mask: tensor of shape ``(batch_size, his_length)``
            bias: tensor of shape ``(batch_size, his_length, num_candidates)``

        Returns:
            A tensor of shape ``(batch_size, num_context_codes, embed_dim)``
        """
        proj = torch.tanh(self.linear(embeddings))
        if bias is None:
            weights = torch.matmul(proj, self.context_codes.T)
        else:
            bias = bias.mean(dim=2).unsqueeze(dim=2)
            weights = torch.matmul(proj, self.context_codes.T) + bias
        temp = weights
        weights = weights.permute(0, 2, 1)
        weights = weights.masked_fill_(~attn_mask.unsqueeze(dim=1), 1e-30)
        weights = F.softmax(weights, dim=2)
        poly_repr = torch.matmul(weights, embeddings)

        return poly_repr, temp, weights


def init_weights(m: nn.Module):
    if isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight.data)

    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias)

    if isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


class AdditiveAttention(nn.Module):
    def __init__(self, dim=100, r=2.):
        super().__init__()
        intermediate = int(dim * r)
        self.attn = nn.Sequential(
            nn.Linear(dim, intermediate),
            nn.Dropout(0.01),
            nn.LayerNorm(intermediate),
            nn.SiLU(),
            nn.Linear(intermediate, 1),
            nn.Softmax(1),
        )
        self.attn.apply(init_weights)

    def forward(self, context):
        """Additive Attention
        Args:
            context (tensor): [B, seq_len, dim] 
        Returns:
            outputs, weights: [B, seq_len, dim], [B, seq_len]
        """
        w = self.attn(context).squeeze(-1)
        return torch.bmm(w.unsqueeze(1), context).squeeze(1)


class MultiHeadMatrixAttention(nn.Module):
    def __init__(self, hidden_dim, head_num):
        super().__init__()
        self.head_num = head_num
        assert hidden_dim % head_num == 0, "hidden_dim {} must divide head_num {}".format(hidden_dim, head_num)
        self.head_dim = hidden_dim // head_num
        self.hidden_dim = hidden_dim

        self.keyProject = nn.Linear(hidden_dim, self.head_dim * head_num)
        self.valueProject = nn.Linear(hidden_dim, self.head_dim * head_num)
        self.queryProject = nn.Linear(hidden_dim, self.head_dim * head_num)

        nn.init.xavier_normal_(self.keyProject.weight)
        nn.init.xavier_normal_(self.valueProject.weight)
        nn.init.xavier_normal_(self.queryProject.weight)

        self.attentionMatrix = nn.Parameter(torch.zeros((self.head_num, self.head_dim, self.head_dim), requires_grad=True))
        nn.init.xavier_normal_(self.attentionMatrix)
    
    def transpose_for_scores(self, x):
        """
        transpose the head_num dimension, to make every head operates in parallel
        """
        # [B, seq_len, head_num, -1]
        new_x_shape = x.size()[:-1] + (self.head_num, -1)
        x = x.view(*new_x_shape)
        return x

    def forward(self, key, value, query, attention_mask=None):
        """ customized bert self attention, attending to the references

        Args:
            hidden_states: normally encoded candidate news, [batch_size, signal_length, hidden_dim]

        Returns:
            attn_output: [batch_size, signal_length, value_dim * num_head]
        """
        # [batch_size, head_num, *, head_dim]
        query = self.transpose_for_scores(self.queryProject(query))
        key = self.transpose_for_scores(self.keyProject(key))
        value = self.transpose_for_scores(self.valueProject(value))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # [B, N, M, D] * [M, D, D] -> [B, N, M, D]
        print(query.shape, self.attentionMatrix.shape)
        query = query.view((-1, self.head_num, self.head_dim))
        attention_scores = torch.matmul(query, self.attentionMatrix)
        query = query.view(key.shape)
        attention_scores = torch.matmul(attention_scores, key.transpose(-1, -2))   # [B, head_num, seq_len, seq_len]

        # [bs, hn, sl+1, *]
        attention_scores = (attention_scores / math.sqrt(self.head_dim))
        if attention_mask is not None:
            attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        else:
            attention_probs = torch.softmax(attention_scores, -1)

        attn_output = torch.matmul(attention_probs, value)

        # [batch_size, signal_length, head_num, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        new_shape = attn_output.size()[:-2] + (self.head_dim * self.head_num,)
        attn_output = attn_output.view(*new_shape)

        return attn_output

