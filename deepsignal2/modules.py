#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Filename:
# @Date: 2018-12-05 16:30
# @author: huangneng
# @contact: huangneng@csu.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

PAD = 0
vocab_size = 1024
embedding_size = 128


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & nn.Parameter(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data), requires_grad=False)
    return tgt_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -float('inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class multiheadattention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(multiheadattention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        # nn.init.xavier_normal_(self.w_q.weight)
        # nn.init.xavier_normal_(self.w_k.weight)
        # nn.init.xavier_normal_(self.w_v.weight)
        # nn.init.xavier_normal_(self.fc.weight)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        query = self.w_q(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        output, self.attn = attention(query=query, key=key, value=value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        output = output.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        output = self.fc(output)

        return output, self.attn


class FFN(nn.Module):
    # feedforward layer
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.w_2 = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.w_2(inter)
        return output + x


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
        x = x + nn.Parameter(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = multiheadattention(h=n_head, d_model=d_model, dropout=dropout)
        self.ffn = FFN(input_size=d_model, hidden_size=d_ff, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_dropout = nn.Dropout(dropout)

    def forward(self, signal_emb, src_mask):
        input = signal_emb
        input_norm = self.layer_norm(signal_emb)
        enc_out, enc_self_attn = self.slf_attn(input_norm, input_norm, input_norm, src_mask)
        enc_out = input + self.norm_dropout(enc_out)

        enc_out = self.ffn(enc_out)

        return enc_out, enc_self_attn


def _get_lout(lin):
    lout = lin
    for i in range(3):
        lout = math.floor(float(lout-1)/2 + 1)
    return int(lout)


class SignalEncoderClassifier(nn.Module):
    def __init__(self, signal_len, d_model, d_ff, n_head, num_encoder_layers, num_classes=2, dropout=0.1):
        super(SignalEncoderClassifier, self).__init__()
        self.src_embed = nn.Sequential(nn.Conv1d(in_channels=1,
                                                 out_channels=d_model//2,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=d_model//2),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                                       nn.Conv1d(in_channels=d_model//2,
                                                 out_channels=d_model,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=d_model),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                                       nn.Conv1d(in_channels=d_model,
                                                 out_channels=d_model,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=d_model),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        # TODO: why padding_idx=0
        self.position_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)

        self.stack_layers = nn.ModuleList([EncoderLayer(d_model=d_model, d_ff=d_ff,
                                                        n_head=n_head,
                                                        dropout=dropout) for _ in range(num_encoder_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=dropout)
        # self.fc1 = nn.Linear(d_model * _get_lout(signal_len), _get_lout(signal_len))
        # self.fc2 = nn.Linear(_get_lout(signal_len), num_classes)
        self.fc = nn.Linear(d_model * _get_lout(signal_len), num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, signal):
        # param signal: shape (batch, signal_len, feature_num)
        signal = signal.reshape(-1, signal.size(1), 1).float()
        # print(signal.size())
        signal = signal.transpose(-1, -2)  # (N,C,L)
        embed_out = self.src_embed(signal)  # (N,C,L)
        # print("embed_out.size: {}".format(embed_out.size()))
        embed_out = embed_out.transpose(-1, -2)  # (N,L,C)
        # print("embed_out.size2: {}".format(embed_out.size()))
        enc_output = self.position_encoding(embed_out)
        # print("enc_output.size1: {}".format(enc_output.size()))
        src_mask = torch.zeros(enc_output.size(0), 1, enc_output.size(1), dtype=torch.uint8).to(enc_output.device)
        for layer in self.stack_layers:
            enc_output, enc_slf_attn = layer(enc_output, src_mask)
        enc_output = self.layer_norm(enc_output)
        # print("enc_output.size2: {}".format(enc_output.size()))

        # fc_out = self.fc1(enc_output.reshape(enc_output.size(0), -1))
        # fc_out = self.relu(fcout)
        # fc_out = self.dropout(fcout)
        # fc_out = self.fc2(fcout)

        fc_out = self.fc(enc_output.reshape(enc_output.size(0), -1))
        return fc_out, self.sigmoid(fc_out)


class SeqEncoderClassifier(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, n_head, num_encoder_layers, num_classes=2, dropout=0.1):
        super(SeqEncoderClassifier, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.seq_len = seq_len
        self.src_embed = nn.Sequential(nn.Conv1d(in_channels=131,
                                                 out_channels=d_model//2,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=d_model//2),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                       nn.Conv1d(in_channels=d_model//2,
                                                 out_channels=d_model,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=d_model),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                       nn.Conv1d(in_channels=d_model,
                                                 out_channels=d_model,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=d_model),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3, stride=1, padding=1))
        # TODO: why padding_idx=0
        self.position_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)

        self.stack_layers = nn.ModuleList([EncoderLayer(d_model=d_model, d_ff=d_ff,
                                                        n_head=n_head,
                                                        dropout=dropout) for _ in range(num_encoder_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=dropout)
        # self.fc1 = nn.Linear(d_model * seq_len, seq_len)
        # self.fc2 = nn.Linear(seq_len, num_classes)
        self.fc = nn.Linear(d_model * seq_len, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, kmer, base_means, base_stds, base_signal_lens):
        # param signal: shape (batch, signal_len, feature_num)
        kmer_embed = self.embed(kmer.long())
        base_means = torch.reshape(base_means, (-1, self.seq_len, 1)).float()
        base_stds = torch.reshape(base_stds, (-1, self.seq_len, 1)).float()
        base_signal_lens = torch.reshape(base_signal_lens, (-1, self.seq_len, 1)).float()
        seqs = torch.cat((kmer_embed, base_means, base_stds, base_signal_lens), 2)
        # print(seqs.size())
        seqs = seqs.transpose(-1, -2)  # (N,C,L)
        # print(seqs.size())
        embed_out = self.src_embed(seqs)  # (N,C,L)
        # print("embed_out.size: {}".format(embed_out.size()))
        embed_out = embed_out.transpose(-1, -2)  # (N,L,C)
        # print("embed_out.size2: {}".format(embed_out.size()))
        enc_output = self.position_encoding(embed_out)
        # print("enc_output.size1: {}".format(enc_output.size()))
        src_mask = torch.zeros(enc_output.size(0), 1, enc_output.size(1), dtype=torch.uint8).to(enc_output.device)
        for layer in self.stack_layers:
            enc_output, enc_slf_attn = layer(enc_output, src_mask)
        enc_output = self.layer_norm(enc_output)
        # print("enc_output.size2: {}".format(enc_output.size()))

        # fc_out = self.fc1(enc_output.reshape(enc_output.size(0), -1))
        # fc_out = self.relu(fc_out)
        # fc_out = self.dropout(fc_out)
        # fc_out = self.fc2(fc_out)

        fc_out = self.fc(enc_output.reshape(enc_output.size(0), -1))

        return fc_out, self.sigmoid(fc_out)
