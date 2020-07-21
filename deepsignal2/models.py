#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.autograd as autograd
import math

import torch.utils
import torch.utils.checkpoint

from utils.constants_torch import use_cuda

tcheckpoint = torch.utils.checkpoint.checkpoint
checkpoint = lambda f, *args, **kwargs: f(*args, **kwargs)


def prepare_model_data(abatch, sequence_length, nnembedding):
    sampleinfo, kmer, base_means, base_stds, base_signal_lens, cent_signals, label = abatch
    kmer_embed = nnembedding(kmer.long())
    base_means = torch.reshape(base_means, (-1, sequence_length, 1)).float()
    base_stds = torch.reshape(base_stds, (-1, sequence_length, 1)).float()
    base_signal_lens = torch.reshape(base_signal_lens, (-1, sequence_length, 1)).float()
    brnn_feed = torch.cat((kmer_embed, base_means, base_stds, base_signal_lens), 2)
    # return sampleinfo, brnn_feed, one_hot_embedding(label, num_classes).long()
    return sampleinfo, brnn_feed, label


# BRNN model ==========================================================
# Bidirectional recurrent neural network, LSTM (many-to-one)
class SeqBiLSTM(nn.Module):
    def __init__(self, seq_len=11, num_layers=3, num_classes=2,
                 dropout_rate=0.5, hidden_size=256, vocab_size=16,
                 embedding_size=4, is_base=True, is_signallen=True):
        super(SeqBiLSTM, self).__init__()
        self.model_type = 'BiLSTM'
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base
        self.is_base = is_base
        self.is_signallen = is_signallen
        self.sigfea_num = 3 if self.is_signallen else 2
        if is_base:
            self.lstm = nn.LSTM(embedding_size+self.sigfea_num, hidden_size, num_layers,
                                dropout=dropout_rate, batch_first=True, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.sigfea_num, hidden_size, num_layers,
                                dropout=dropout_rate, batch_first=True, bidirectional=True)

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # 2 for bidirection
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def init_hidden(self, batch_size):
        # Set initial states
        # h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)  # 2 for bidirection
        # c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
        h0 = autograd.Variable(torch.randn(self.num_layers * 2, batch_size, self.hidden_size))
        c0 = autograd.Variable(torch.randn(self.num_layers * 2, batch_size, self.hidden_size))
        if use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return h0, c0

    def forward(self, kmer, base_means, base_stds, base_signal_lens):
        base_means = torch.reshape(base_means, (-1, self.seq_len, 1)).float()
        base_stds = torch.reshape(base_stds, (-1, self.seq_len, 1)).float()
        base_signal_lens = torch.reshape(base_signal_lens, (-1, self.seq_len, 1)).float()
        if self.is_base:
            kmer_embed = self.embed(kmer.long())
            if self.is_signallen:
                brnn_feed = torch.cat((kmer_embed, base_means, base_stds, base_signal_lens), 2)  # (N, L, C)
            else:
                brnn_feed = torch.cat((kmer_embed, base_means, base_stds), 2)  # (N, L, C)
        else:
            if self.is_signallen:
                brnn_feed = torch.cat((base_means, base_stds, base_signal_lens), 2)  # (N, L, C)
            else:
                brnn_feed = torch.cat((base_means, base_stds), 2)  # (N, L, C)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # out, (hn, cn) = self.lstm(brnn_feed, self.init_hidden(brnn_feed.size(0)))
        out, _ = self.lstm(brnn_feed, self.init_hidden(brnn_feed.size(0)))

        # out = out[:, -1, :]
        out_fwd_last = out[:, -1, :self.hidden_size]
        out_bwd_last = out[:, 0, self.hidden_size:]
        # print(hn[-1].eq(out_bwd_last))
        # print(hn[-2].eq(out_fwd_last))
        out = torch.cat((out_fwd_last, out_bwd_last), 1)

        out = self.dropout1(out)

        # Decode the hidden state of the last time step
        out = self.fc1(out)
        out = self.dropout2(out)
        out = self.tanh(out)
        out = self.fc2(out)
        # print(out.size())

        return out, self.softmax(out)


# Transformer Encoder ========================================
# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SeqTransformer(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""
    def __init__(self, seq_len=11, num_layers=3, num_classes=2,
                 dropout_rate=0.5, d_model=256, nhead=4, nhid=512,
                 nvocab=16, nembed=4, is_base=True, is_signallen=True):
        super(SeqTransformer, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.is_base = is_base
        self.is_signallen = is_signallen
        self.sigfea_num = 3 if self.is_signallen else 2

        self.d_model = d_model

        if is_base:
            self.embed = nn.Embedding(nvocab, nembed)
            self.src_embed = nn.Sequential(nn.Conv1d(in_channels=nembed+self.sigfea_num,
                                                     out_channels=self.d_model // 2,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     bias=False),
                                           nn.BatchNorm1d(num_features=self.d_model // 2),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                           nn.Conv1d(in_channels=self.d_model // 2,
                                                     out_channels=self.d_model,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     bias=False),
                                           nn.BatchNorm1d(num_features=self.d_model),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                           nn.Conv1d(in_channels=self.d_model,
                                                     out_channels=self.d_model,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     bias=False),
                                           nn.BatchNorm1d(num_features=self.d_model),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool1d(kernel_size=3, stride=1, padding=1))
        else:
            self.src_embed = nn.Sequential(nn.Conv1d(in_channels=self.sigfea_num,
                                                     out_channels=self.d_model // 2,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     bias=False),
                                           nn.BatchNorm1d(num_features=self.d_model // 2),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                           nn.Conv1d(in_channels=self.d_model // 2,
                                                     out_channels=self.d_model,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     bias=False),
                                           nn.BatchNorm1d(num_features=self.d_model),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                           nn.Conv1d(in_channels=self.d_model,
                                                     out_channels=self.d_model,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     bias=False),
                                           nn.BatchNorm1d(num_features=self.d_model),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool1d(kernel_size=3, stride=1, padding=1))

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.d_model, dropout_rate)
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, nhid, dropout_rate)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.decoder1 = nn.Linear(self.d_model * self.seq_len, self.d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.tanh = nn.Tanh()
        self.decoder2 = nn.Linear(self.d_model, num_classes)
        self.softmax = nn.Softmax(1)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, kmer, base_means, base_stds, base_signal_lens, has_mask=True):
        base_means = torch.reshape(base_means, (-1, self.seq_len, 1)).float()
        base_stds = torch.reshape(base_stds, (-1, self.seq_len, 1)).float()
        base_signal_lens = torch.reshape(base_signal_lens, (-1, self.seq_len, 1)).float()
        if self.is_base:
            kmer_embed = self.embed(kmer.long())
            if self.is_signallen:
                out = torch.cat((kmer_embed, base_means, base_stds, base_signal_lens), 2)  # (N, L, C)
            else:
                out = torch.cat((kmer_embed, base_means, base_stds), 2)  # (N, L, C)
        else:
            if self.is_signallen:
                out = torch.cat((base_means, base_stds, base_signal_lens), 2)  # (N, L, C)
            else:
                out = torch.cat((base_means, base_stds), 2)  # (N, L, C)

        out = out.transpose(-1, -2)  # (N, C, L)
        out = self.src_embed(out)  # (N, C, L)
        out = out.transpose(-1, -2)  # (N, L, C)
        out = out.transpose(0, 1)  # (L, N, C)
        out = self.pos_encoder(out)  # (L, N, C)

        if has_mask:
            device = out.device
            if self.src_mask is None or self.src_mask.size(0) != len(out):
                mask = self._generate_square_subsequent_mask(len(out)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        out = self.transformer_encoder(out, self.src_mask)  # (L, N, C)
        out = out.transpose(0, 1)  # (N, L, C)
        out = out.reshape(out.size(0), -1)

        # output logits
        out = self.decoder1(out)
        out = self.dropout(out)
        out = self.tanh(out)
        out = self.decoder2(out)
        return out, self.softmax(out)
