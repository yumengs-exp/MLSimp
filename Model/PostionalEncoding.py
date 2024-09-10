import torch.nn as nn
import torch
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
        [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table)           # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs,tindex):                                         # enc_inputs: [batch_size, seq_len, d_model]
        tindex = tindex-tindex[0]
        enc_inputs += self.pos_table[tindex, :]
        return self.dropout(enc_inputs)