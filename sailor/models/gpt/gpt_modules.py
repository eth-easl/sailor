# adapt from https://github.com/DS3Lab/Decentralized_FM_alpha/blob/main/modules/gpt_modules.py

import math
import torch
from torch import nn
from torch.nn import functional

# this function is adapted from https://github.com/DS3Lab/CocktailSGD/blob/master/modules/utils.py


def gpt_loss_func(inp, target):
    lm_logits, labels = inp, target
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, head_num):
        super().__init__()
        # in Attention: model_dim=768 (nx=n_embd)
        assert model_dim % head_num == 0
        self.model_dim = model_dim
        self.head_num = head_num
        self.split_size = model_dim // head_num
        self.q_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.scale = math.sqrt(self.split_size)

        # self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(model_dim, model_dim)

    def forward(self, inp):
        bs = inp.size(0)
        # perform linear operation and split into N heads
        k = self.k_linear(inp).view(bs, -1, self.head_num, self.split_size)
        q = self.q_linear(inp).view(bs, -1, self.head_num, self.split_size)
        v = self.v_linear(inp).view(bs, -1, self.head_num, self.split_size)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        scores = functional.softmax(scores, dim=-1)
        scores = torch.matmul(scores, v)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(
            bs, -1, self.model_dim)
        output = self.out(concat)
        return output + inp  # Put residual connection here.


class TwoLayerMLP(nn.Module):
    def __init__(self, model_dim, feedford_dim):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, feedford_dim)
        self.linear2 = nn.Linear(feedford_dim, model_dim)

    def forward(self, inp):
        a1 = functional.relu(self.linear1(inp))
        a2 = self.linear2(a1)
        return inp + a2


class GPTTransformerLayer(nn.Module):
    def __init__(self, model_dim, head_num, feedforward_dim=2048, layer_norm_eps=1e-5) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(model_dim, head_num)
        # Implementation of Feedforward model
        self.mlp = TwoLayerMLP(model_dim, feedforward_dim)
        self.norm1 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        # x = x + self.dropout_1(self.attn(x2, x2, x2))
        x = self.attn(x)
        x = self.norm2(x)
        # x = x + self.dropout_2(self.ff(x2))
        x = self.mlp(x)
        return x


def get_position_id(seq_length, size_input, device):
    return torch.arange(seq_length, device=device).unsqueeze(0).expand(size_input, seq_length)


class GPTEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.
    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, vocab_size, embedding_dim, seq_length, num_token_types=0):
        super().__init__()
        # Keep the input dimensions.
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_token_types = num_token_types

        self.vocab_embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=None,
                                                  max_norm=None,  norm_type=2, scale_grad_by_freq=False, sparse=False)
        torch.nn.init.xavier_normal_(self.vocab_embedding.weight)
        self.position_embedding = torch.nn.Embedding(seq_length, embedding_dim)
        torch.nn.init.xavier_normal_(self.position_embedding.weight)
        if num_token_types > 0:
            self.token_type_embedding = torch.nn.Embedding(
                num_token_types, embedding_dim)
        else:
            self.token_type_embedding = None

    def forward(self, input_ids, position_ids=None, tokentype_ids=None):
        word_embeddings = self.vocab_embedding(input_ids)
        if position_ids is None:
            position_ids = get_position_id(
                self.seq_length, word_embeddings.shape[0], word_embeddings.device)
        pos_embeddings = self.position_embedding(position_ids)
        embeddings = word_embeddings + pos_embeddings
        if tokentype_ids:
            assert self.token_type_embedding is not None
            embeddings = embeddings + self.token_type_embedding(tokentype_ids)
        return embeddings


class GPTLMHead(nn.Module):
    def __init__(self, embedding_dim, vocab_size, layer_norm_eps=1e-5):
        super().__init__()
        self.ln_f = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x
