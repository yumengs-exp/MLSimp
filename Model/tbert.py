import math
from itertools import zip_longest

import numpy as np
import torch
from torch import nn
from sklearn.utils import shuffle

from Utils.tbert_utils import next_batch, weight_init
import datetime

def gen_random_mask(src_valid_lens, src_len, mask_prop):
    """
    @param src_valid_lens: valid length of sequence, shape (batch_size)
    """
    # all_index = np.arange((batch_size * src_len)).reshape(batch_size, src_len)
    # all_index = shuffle_along_axis(all_index, axis=1)
    # mask_count = math.ceil(mask_prop * src_len)
    # masked_index = all_index[:, :mask_count].reshape(-1)
    # return masked_index
    index_list = []
    for batch, l in enumerate(src_valid_lens):
        mask_count = torch.ceil(mask_prop * l).int()
        masked_index = torch.randperm(l)[:mask_count]
        masked_index += src_len * batch
        index_list.append(masked_index)
    return torch.cat(index_list).long().to(src_valid_lens.device)


def gen_casual_mask(seq_len, include_self=True):
    """
    Generate a casual mask which prevents i-th output element from
    depending on any input elements from "the future".
    Note that for PyTorch Transformer model, sequence mask should be
    filled with -inf for the masked positions, and 0.0 else.

    :param seq_len: length of sequence.
    :return: a casual mask, shape (seq_len, seq_len)
    """
    if include_self:
        mask = 1 - torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
    else:
        mask = 1 - torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1)
    return mask.bool()


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len):
        super().__init__()
        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, **kwargs):
        return self.pe[:, :x.size(1)]


class TemporalEncoding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1. / embed_size)

    def forward(self, x, **kwargs):
        timestamp = kwargs['timestamp']  # (batch, seq_len)
        time_encode = timestamp.unsqueeze(-1) * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1)
        time_encode = torch.cos(time_encode)
        return self.div_term * time_encode


class TBERTEmbedding(nn.Module):
    def __init__(self, encoding_layer, embed_size, num_vocab,grid_emb_matrix=None):
        super().__init__()
        self.embed_size = embed_size
        self.num_vocab = num_vocab
        self.encoding_layer = encoding_layer
        self.add_module('encoding', self.encoding_layer)

        self.token_embed = nn.Embedding(num_vocab, embed_size, padding_idx=0)
        if grid_emb_matrix!=None:
            self.token_embed.weight = nn.Parameter(grid_emb_matrix)
        # self.token_embed.weight.data.uniform_(-0.5/embed_size, 0.5/embed_size)

    def forward(self, x, **kwargs):
        token_embed = self.token_embed(x)
        pos_embed = self.encoding_layer(x, **kwargs)
        return token_embed + pos_embed


class TBERT(nn.Module):
    def __init__(self, embed, hidden_size, num_layers, num_heads, init_param=False, detach=True):
        super().__init__()
        self.embed_size = embed.embed_size
        self.num_vocab = embed.num_vocab

        self.embed = embed
        self.add_module('embed', embed)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=num_heads,
                                                   dim_feedforward=hidden_size, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(self.embed_size, eps=1e-6))
        self.detach = detach
        if init_param:
            self.apply(weight_init)

    def forward(self, x, **kwargs):
        """
        @param x: sequence of tokens, shape (batch, seq_len).
        """
        # seq_len = x.size(1)
        # downstream = kwargs.get('downstream', False)

        src_key_padding_mask = (x == 0)
        token_embed = self.embed(x, **kwargs)  # (batch_size, seq_len, embed_size)

        src_mask = None

        encoder_out = self.encoder(token_embed.transpose(0, 1), mask=src_mask,
                                   src_key_padding_mask=src_key_padding_mask).transpose(0, 1)  # (batch_size, src_len, embed_size)
        if self.detach:
            encoder_out = encoder_out.detach()
        return encoder_out

    def static_embed(self):
        return self.embed.token_embed.weight[:self.num_vocab].detach().cpu().numpy()


class MaskedLM(nn.Module):
    def __init__(self, input_size, vocab_size):
        super().__init__()
        self.linear = nn.Linear(input_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.CrossEntropyLoss()

        self.vocab_size = vocab_size

    def forward(self, x, **kwargs):
        """
        :param x: input sequence (batch, seq_len, embed_size).
        :param origin_tokens: original tokens, shape (batch, seq_len)
        :return: the loss value of MLM objective.
        """
        origin_tokens = kwargs['origin_tokens']
        origin_tokens = origin_tokens.reshape(-1)
        lm_pre = self.linear(self.dropout(x))  # (batch, seq_len, vocab_size)
        lm_pre = lm_pre.reshape(-1, self.vocab_size)  # (batch * seq_len, vocab_size)
        return self.loss_func(lm_pre, origin_tokens)







def train_tbert(dataset, tbert_model:TBERT, obj_models, mask_prop, num_epoch, batch_size, device,save_path,obj_save_path):
    tbert_model = tbert_model.to(device)
    obj_models = obj_models.to(device)
    src_tokens, src_ts, src_lens = zip(*dataset.gen_sequence(select_days=0))

    optimizer = torch.optim.Adam(list(tbert_model.parameters()) + list(obj_models.parameters()), lr=1e-4)
    cnt = 0
    for epoch in range(num_epoch):
        for batch in next_batch(shuffle(list(zip(src_tokens, src_ts, src_lens))), batch_size=batch_size):
            # Value filled with num_loc stands for masked tokens that shouldn't be considered.
            src_batch, src_t_batch, src_len_batch = zip(*batch)
            src_batch = np.transpose(np.array(list(zip_longest(*src_batch, fillvalue=0))))
            src_t_batch = np.transpose(np.array(list(zip_longest(*src_t_batch, fillvalue=0))))

            src_batch = torch.tensor(src_batch).long().to(device)
            src_t_batch = torch.tensor(src_t_batch).float().to(device)
            hour_batch = (src_t_batch % (24 * 60 * 60) / 60 / 60).long()

            batch_len, src_len = src_batch.size(0), src_batch.size(1)
            src_valid_len = torch.tensor(src_len_batch).long().to(device)

            mask_index = gen_random_mask(src_valid_len, src_len, mask_prop=mask_prop)

            src_batch = src_batch.reshape(-1)
            hour_batch = hour_batch.reshape(-1)
            origin_tokens = src_batch[mask_index]  # (num_masked)
            origin_hour = hour_batch[mask_index]

            # Value filled with num_loc+1 stands for special token <mask>.
            masked_tokens = src_batch.index_fill(0, mask_index, 1).reshape(batch_len, -1)  # (batch_size, src_len)

            tbert_out = tbert_model(masked_tokens, timestamp=src_t_batch)  # (batch_size, src_len, embed_size)
            masked_out = tbert_out.reshape(-1, tbert_model.embed_size)[mask_index]  # (num_masked, embed_size)
            loss = 0.
            for obj_model in obj_models:
                loss += obj_model(masked_out, origin_tokens=origin_tokens, origin_hour=origin_hour)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'T-Bert training loss:{loss.item()}')
            cnt+=1
            if cnt%100 ==0:
                print('Save Bert model...')
                torch.save(tbert_model.state_dict(), save_path)
                torch.save(obj_model.state_dict(), obj_save_path)

    return tbert_model

