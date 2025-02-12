import torch
import torch.nn as nn
import math


def scaled_attention(query, key, value, attn_mask=None):
    """ calculate scaled attended output of values

    Args:
        query: tensor of [batch_size, *, query_num, key_dim]
        key: tensor of [batch_size, *, key_num, key_dim]
        value: tensor of [batch_size, *, key_num, value_dim]
        attn_mask: tensor of [batch_size, *, query_num, key_num]
    Returns:
        attn_output: tensor of [batch_size, *, query_num, value_dim]
    """

    # make sure dimension matches
    assert query.shape[-1] == key.shape[-1]
    key = key.transpose(-2, -1)

    attn_score = torch.matmul(query, key) / math.sqrt(query.shape[-1])
    attn_prob = torch.softmax(attn_score, -1)
    attn_output = torch.matmul(attn_prob, value)
    return attn_output


def get_attn_mask(attn_mask):
    """
    extend the attention mask

    Args:
        attn_mask: [batch_size, *]

    Returns:
        attn_mask: [batch_size, 1, *, *]
    """
    if attn_mask is None:
        return None

    assert attn_mask.dim() == 2

    extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
    extended_attn_mask2 = extended_attn_mask.squeeze(-2).unsqueeze(-1)

    attn_mask = extended_attn_mask * extended_attn_mask2

    return attn_mask


class AttentionPooler(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.query_news = nn.Parameter(torch.randn(1, manager.hidden_size))
        nn.init.xavier_normal_(self.query_news)

    def forward(self, news_reprs, his_mask=None, *args, **kargs):
        """
        encode user history into a representation vector

        Args:
            news_reprs: batch of news representations, [batch_size, *, hidden_dim]

        Returns:
            user_repr: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        if his_mask is not None:
            his_mask = his_mask.to(news_reprs.device).transpose(-1, -2)
        user_repr = scaled_attention(self.query_news, news_reprs, news_reprs,
                                        attn_mask=his_mask)
        return user_repr


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output