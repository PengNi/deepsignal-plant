#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import random
import numpy as np
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

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


def attention(query, key, value, attn_mask=None, need_weights=True, dropout=None):
    # https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html
    # Needs [batch, heads, seqlen, hid]

    batch_size, heads, query_len, dim = query.size()
    key_len = key.size(2)

    # Scaling by dim due to http://nlp.seas.harvard.edu/2018/04/03/attention.html
    attention_scores = torch.matmul(query, key.transpose(-1, -2).contiguous()) / math.sqrt(dim)
    if attn_mask is not None:
        attn_mask = attn_mask.view(1, 1, *attn_mask.shape[-2:])
        attention_scores = attention_scores + attn_mask # Mask is additive and contains -Infs

    attention_weights = F.softmax(attention_scores, dim=-1)
    if dropout:
        attention_weights = dropout(attention_weights)
    attention_weights = attention_weights.view(batch_size, heads, query_len, key_len)

    mix = torch.matmul(attention_weights, value)
    return mix, attention_weights


class Overparam(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.l1 = nn.Linear(nhid, 2 * nhid)
        #self.l2 = nn.Linear(2 * nhid, 2 * nhid)
        self.inner_act = torch.tanh # GELU()
        self.nhid = nhid

    def forward(self, x):
        c, f = self.l1(x).split(self.nhid, dim=-1)
        #c, f = self.l2(self.inner_act(self.l1(x))).split(self.nhid, dim=-1)
        return torch.sigmoid(f) * torch.tanh(c)


class Attention(nn.Module):
    def __init__(self, nhid, q=True, k=False, v=False, r=False, heads=1, dropout=None):
        super().__init__()
        self.qs = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.ks = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.vs = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.qkvs = nn.Parameter(torch.zeros(size=(1, 3, nhid), dtype=torch.float))
        self.heads = heads
        self.nhid = nhid
        assert nhid % self.heads == 0, 'Heads must divide vector evenly'
        self.drop = nn.Dropout(dropout) if dropout else None
        self.gelu = GELU()
        self.q = nn.Linear(nhid, nhid) if q else None
        self.qln = LayerNorm(nhid, eps=1e-12)
        self.k = nn.Linear(nhid, nhid) if k else None
        self.v = nn.Linear(nhid, nhid) if v else None
        self.r = nn.Linear(2 * nhid, nhid) if r else None
        self.r_gate = nn.Parameter(torch.ones(size=(1, 1, nhid), dtype=torch.float))
        self.vq = None
        self.vq = Overparam(nhid)
        #from fastai.text.models import QRNNLayer
        #self.vq = QRNNLayer(input_size=nhid, hidden_size=nhid, save_prev_x=False, zoneout=0, window=1, output_gate=False, batch_first=False)
        self.vq_collapsed = False

    def vq_collapse(self):
        vs = torch.sigmoid(self.vs)
        #vs, _ = self.vq(vs)
        vs = self.vq(vs)
        self.vs.data = vs.data
        self.vq = None
        self.vq_collapsed = True

    def forward(self, query, key, value, attn_mask=None, batch_first=False, **kwargs):
        # tanh on the value allows us to flip the polarity of the output, helping use the full range
        # Discovered accidentally when I used QRNN_with_tanh_output(sigmoid(vs))
        #qs, ks, vs = torch.sigmoid(self.qs), torch.sigmoid(self.ks), self.vs
        qs, ks, vs = torch.sigmoid(self.qs), torch.sigmoid(self.ks), torch.sigmoid(self.vs)
        #qs, ks, vs = self.qs, self.ks, self.vs
        #vs = torch.tanh(self.vs)
        if self.vq:
            #vs, _ = self.vq(vs)
            vs = self.vq(vs)
            #qs, ks, vs = [x.reshape((1, 1, -1)) for x in self.vq(torch.sigmoid(self.qkvs))[0, :]]
        elif self.vq_collapsed:
            vs = self.vs
        #qs, ks, vs = self.qs, self.ks, self.vs
        #q = qs * query
        #if self.q: query = self.q(query)
        if self.q:
            query = self.q(query)
            query = self.qln(query.float())
        if self.k: key = self.k(key)
        if self.v: value = self.v(value)
        # This essentially scales everything to zero to begin with and then learns from there
        #q, k, v = self.qs * query, self.ks * key, self.vs * value
        q, k, v = qs * query, ks * key, vs * value
        #q, k, v = query, key, vs * value
        #q, k, v = qs * query, ks * key, value
        #k, v = ks * key, vs * value
        #q, k, v = query, key, value
        if self.drop:
            # We won't apply dropout to v as we can let the caller decide if dropout should be applied to the output
            # Applying dropout to q is equivalent to the same mask on k as they're "zipped"
            #q, k, v = self.drop(q), k, v
            q, k, v = self.drop(q), k, self.drop(v)

        original_q = q

        if not batch_first:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        batch_size, query_len, nhid = q.size()
        assert nhid == self.nhid
        key_len = k.size(1)
        ###
        dim = self.nhid // self.heads
        q = q.view(batch_size, query_len, self.heads, dim).transpose(1, 2)
        k, v = [vec.view(batch_size, key_len, self.heads, dim).transpose(1, 2) for vec in [k, v]]

        mix, focus = attention(q, k, v, dropout=self.drop, attn_mask=attn_mask, **kwargs)
        mix = mix.transpose(1, 2).contiguous().view(batch_size, -1, self.nhid)
        if not batch_first:
            mix = mix.transpose(0, 1)

        if self.r:
            # The result should be transformed according to the query
            r = torch.cat([mix, original_q], dim=-1)
            if self.drop: r = self.drop(r)
            r = self.gelu(self.r(r))
            mix = torch.sigmoid(self.r_gate) * mix + r
            # BUG: This does _nothing_ as mix isn't set to r ...
            # But ... I got good results with this ... so ...
            # Let's leave it as is for right now ...
            # This does imply that I don't necessarily need complex post mixing ops

        return mix, focus


class PyTorchAttention(nn.Module):
    def __init__(self, nhid, q=True, k=False, v=False, heads=1, dropout=None):
        super().__init__()
        self.mha = nn.MultiheadAttention(nhid, heads, dropout=dropout)

    def forward(self, q, k, v, attn_mask=None):
        return self.mha(q, k, v, attn_mask=attn_mask)


class Block(nn.Module):
    def __init__(self, embed_dim, hidden_dim, heads=1, dropout=None, rnn=False, residual=True, use_attn=True):
        super().__init__()
        #self.attn = PyTorchAttention(embed_dim, heads=heads, dropout=dropout)
        self.attn = None
        if use_attn:
            self.attn = Attention(embed_dim, heads=heads, r=False, dropout=dropout)
        self.ff = Boom(embed_dim, hidden_dim, dropout=dropout, shortcut=True)
        self.lnstart = LayerNorm(embed_dim, eps=1e-12)
        self.lnmid = LayerNorm(embed_dim, eps=1e-12)
        self.lnmem = LayerNorm(embed_dim, eps=1e-12)
        self.lnout = LayerNorm(embed_dim, eps=1e-12)
        self.lnff = LayerNorm(embed_dim, eps=1e-12)
        self.lnxff = LayerNorm(embed_dim, eps=1e-12)
        self.drop = nn.Dropout(dropout)
        self.gelu = GELU()
        self.residual = residual

        self.rnn = None
        if rnn:
            self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=False)
            if rnn not in [True, False]:
                self.rnn = rnn

    def forward(self, h, pe, attn_mask, mem=None, hidden=None):
        new_mem = None

        h = self.lnstart(h)

        if self.rnn:
            x, new_hidden = self.rnn(h, None if hidden is None else hidden)
            #x = self.rnn_down(self.drop(x))

            # Trim the end off if the size is different
            ninp = h.shape[-1]
            z = torch.narrow(x, -1, 0, x.shape[-1] // ninp * ninp)
            # Divide the hidden size evenly into chunks
            z = x.view(*x.shape[:-1], x.shape[-1] // ninp, ninp)
            # Collapse the chunks through summation
            #h = h + self.drop(x).sum(dim=-2)
            x = self.drop(z).sum(dim=-2)
            #x = x + z.sum(dim=-2)

            h = h + x if self.residual else x.float()

        focus, new_mem = None, []
        if self.attn is not None:
            mh = self.lnmem(h)
            h = self.lnmid(h)

            if mem is not None:
                bigh = torch.cat([mem, mh], dim=0)
            else:
                bigh = mh
            new_mem = bigh[-len(pe):]

            q, k = h, bigh

            x, focus = checkpoint(self.attn, q, k, bigh, attn_mask)
            #x, focus = tcheckpoint(self.attn, q, k, bigh, attn_mask)
            x = self.drop(x)
            h = x + h

        if self.ff:
            h, x = self.lnff(h), self.lnxff(h)
            x = checkpoint(self.ff, x)
            #x = tcheckpoint(self.ff, h)
            x = self.drop(x)
            h = x + h

        return h, new_mem, new_hidden, focus


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        #return torch.nn.functional.gelu(x.float())
        # The first approximation has more operations than the second
        # See https://arxiv.org/abs/1606.08415
        #return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return x * torch.sigmoid(1.702 * x)


class Boom(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, shortcut=False):
        super(Boom, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) if dropout else None
        if not shortcut:
            self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.shortcut = shortcut
        #self.act = nn.ReLU()
        self.act = GELU()
        #self.act = nn.Tanh()

    def forward(self, input):
        x = self.act(self.linear1(input))
        if self.dropout: x = self.dropout(x)
        if self.shortcut:
            # Trim the end off if the size is different
            ninp = input.shape[-1]
            x = torch.narrow(x, -1, 0, x.shape[-1] // ninp * ninp)
            # Divide the hidden size evenly into chunks
            x = x.view(*x.shape[:-1], x.shape[-1] // ninp, ninp)
            # Collapse the chunks through summation
            #h = h + self.drop(x).sum(dim=-2)
            z = x.sum(dim=-2)
        else:
            z = self.linear2(x)

        return z


class SHARNN(nn.Module):
    def __init__(self, seq_len=11, ninp=256, nhid=256, nlayers=4, num_classes=2,
                 dropout=0.5, dropouth=0.5, dropouti=0.5,
                 dropoute=0.1, wdrop=0, tie_weights=False, nvocab=5, nembed=5):
        super().__init__()

        self.seq_len = seq_len
        embed_dim = ninp
        hidden_dim = nhid
        self.ninp, self.nhid = ninp, nhid
        self.nlayers = nlayers
        self.num_max_positions = 2048
        self.num_heads = 1  # 4
        num_layers = nlayers
        self.causal = True
        self.drop = nn.Dropout(dropout)
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)

        self.embed = nn.Embedding(nvocab, nembed)
        self.src_embed = nn.Sequential(nn.Conv1d(in_channels=nembed + 3,
                                                 out_channels=ninp // 2,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=ninp // 2),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                       nn.Conv1d(in_channels=ninp // 2,
                                                 out_channels=ninp,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=ninp),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                       nn.Conv1d(in_channels=ninp,
                                                 out_channels=ninp,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=ninp),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            rnn = True
            self.blocks.append(Block(embed_dim, hidden_dim, self.num_heads,
                                     dropout=dropouth, rnn=rnn, residual=False,
                                     use_attn=True if idx == num_layers - 2 else False))

        self.pos_emb = [0] * self.num_max_positions

        # self.encoder = nn.Embedding(num_embeddings, embed_dim)
        # self.decoder = nn.Linear(embed_dim, num_embeddings)
        # if tie_weights:
        #     self.decoder.weight = self.encoder.weight

        # self.apply(self.init_weights)

        self.fc = nn.Linear(self.seq_len * self.ninp, num_classes)
        self.softmax = nn.Softmax(1)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=0.1 / np.sqrt(self.ninp))

        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, kmer, base_means, base_stds, base_signal_lens,
                hidden=None, mems=None, padding_mask=None):
        """ Input has shape [seq length, batch] """
        # e = self.encoder(x)
        # e = self.idrop(e)

        kmer_embed = self.embed(kmer.long())
        base_means = torch.reshape(base_means, (-1, self.seq_len, 1)).float()
        base_stds = torch.reshape(base_stds, (-1, self.seq_len, 1)).float()
        base_signal_lens = torch.reshape(base_signal_lens, (-1, self.seq_len, 1)).float()
        e = torch.cat((kmer_embed, base_means, base_stds, base_signal_lens), 2)  # (N, L, C)

        e = e.transpose(-1, -2)  # (N, C, L)
        e = self.src_embed(e)  # (N, C, L)
        e = e.transpose(-1, -2)  # (N, L, C)
        e = e.transpose(0, 1)  # (L, N, C)

        if mems is not None:
            maxmem = self.num_max_positions - len(e)
            mems = [m[-maxmem:] for m in mems]

        # total_length = len(x) + (len(mems[0]) if mems else 0)

        pe = self.pos_emb #* 0

        h = e

        new_hidden = []

        new_mems = []

        # focus = []

        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(e), len(e)), -float('Inf'), device=h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)
            if mems:
                max_mems = max(len(m) for m in mems)
                happy = torch.zeros((len(e), max_mems), device=h.device, dtype=h.dtype)
                attn_mask = torch.cat([happy, attn_mask], dim=-1)

        for idx, block in enumerate(self.blocks):
            mem = mems[idx] if mems else None
            hid = hidden[idx] if hidden else None

            h, m, nh, f = block(h, pe, attn_mask=attn_mask, mem=mem, hidden=hid)

            new_hidden.append(nh)
            new_mems.append(m)

        h = self.drop(h)

        h = h.transpose(0, 1)
        h = self.fc(h.reshape(h.size(0), -1))
        hlogits = self.softmax(h)

        return h, hlogits, new_hidden, new_mems
