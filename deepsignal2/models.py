#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math

import torch.utils
import torch.utils.checkpoint
tcheckpoint = torch.utils.checkpoint.checkpoint
#checkpoint = torch.utils.checkpoint.checkpoint
checkpoint = lambda f, *args, **kwargs: f(*args, **kwargs)


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


class SeqEncoderClassifier(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, seq_len=11, d_model=512, nhead=8, nhid=2048, nlayers=6, num_classes=2,
                 dropout=0.5, nvocab=1024, nembed=128):
        super(SeqEncoderClassifier, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'TransformerEncoder'
        self.embed = nn.Embedding(nvocab, nembed)
        self.src_embed = nn.Sequential(nn.Conv1d(in_channels=nembed+3,
                                                 out_channels=d_model // 2,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=d_model // 2),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                       nn.Conv1d(in_channels=d_model // 2,
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

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.seq_len = seq_len
        self.d_model = d_model

        self.fc = nn.Linear(self.d_model * self.seq_len, num_classes)
        self.softmax = nn.Softmax(1)

        # self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, kmer, base_means, base_stds, base_signal_lens, has_mask=True):

        kmer_embed = self.embed(kmer.long())
        base_means = torch.reshape(base_means, (-1, self.seq_len, 1)).float()
        base_stds = torch.reshape(base_stds, (-1, self.seq_len, 1)).float()
        base_signal_lens = torch.reshape(base_signal_lens, (-1, self.seq_len, 1)).float()
        out = torch.cat((kmer_embed, base_means, base_stds, base_signal_lens), 2)  # (N, L, C)

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
        out = out.transpose(0, 1)
        # out = self.fc(out.reshape(out.size(0), -1))
        # return out, self.softmax(out)
        return out.reshape(out.size(0), -1)


class SignalEncoderClassifier(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, signal_len=128, d_model=512, nhead=8, nhid=2048, nlayers=6, num_classes=2,
                 dropout=0.5):
        super(SignalEncoderClassifier, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'TransformerEncoder'
        self.src_embed = nn.Sequential(nn.Conv1d(in_channels=1,
                                                 out_channels=d_model // 2,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=d_model // 2),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                       nn.Conv1d(in_channels=d_model // 2,
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
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.signal_len = signal_len
        self.d_model = d_model

        self.fc = nn.Linear(self.d_model * self.signal_len, num_classes)
        self.softmax = nn.Softmax(1)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, signals, has_mask=True):

        out = signals.reshape(-1, signals.size(1), 1).float()  # (N, L, C)

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
        out = out.transpose(0, 1)
        # out = self.fc(out.reshape(out.size(0), -1))
        # return out, self.softmax(out)
        return out.reshape(out.size(0), -1)


class EncoderClassifier(nn.Module):
    def __init__(self, seq_len=11, signal_len=128, d_model=512, nhead=8, nhid=2048, nlayers=6, num_classes=2,
                 dropout=0.5, nvocab=5, nembed=5, is_seq=True, is_signal=True):
        super(EncoderClassifier, self).__init__()
        assert(is_seq | is_signal is True)
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'TransformerEncoder'
        self.seq_len = seq_len
        self.signal_len = signal_len
        self.d_model = d_model
        self.is_seq = is_seq
        self.is_signal = is_signal

        self.seq_encoder = SeqEncoderClassifier(seq_len, d_model, nhead, nhid, nlayers,
                                                num_classes, dropout, nvocab, nembed)
        self.signal_encoder = SignalEncoderClassifier(signal_len, d_model, nhead, nhid, nlayers,
                                                      num_classes, dropout)

        self.fc_seq = nn.Linear(self.d_model * self.seq_len, num_classes)
        self.fc_signal = nn.Linear(self.d_model * self.signal_len, num_classes)
        self.fc = nn.Linear(self.d_model * (self.signal_len + self.seq_len), num_classes)
        self.softmax = nn.Softmax(1)

    def forward(self, kmer, base_means, base_stds, base_signal_lens, signals, has_mask=True):

        if self.is_seq and not self.is_signal:
            out = self.seq_encoder(kmer, base_means, base_stds, base_signal_lens, has_mask)
            out = self.fc_seq(out)
        elif self.is_signal and not self.is_seq:
            out = self.signal_encoder(signals, has_mask)
            out = self.fc_signal(out)
        else:
            out_seq = self.seq_encoder(kmer, base_means, base_stds, base_signal_lens, has_mask)
            out_signal = self.seq_encoder(kmer, base_means, base_stds, base_signal_lens, has_mask)
            out = torch.cat((out_seq, out_signal), 1)
            out = self.fc(out)
        return out, self.softmax(out)
