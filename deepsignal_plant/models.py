#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import torch.utils
import torch.utils.checkpoint

from .utils.constants_torch import use_cuda


# inner module ================================================
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''


class BasicBlock(nn.Module):
    """use Conv1d and BatchNorm1d"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_3layers(nn.Module):
    """Conv1d"""

    def __init__(self, block, num_blocks, strides, out_channels=128, init_channels=1, in_planes=4):
        super(ResNet_3layers, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv1d(init_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_planes)
        # three group of blocks
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=strides[0])
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=strides[1])
        self.layer3 = self._make_layer(block, out_channels, num_blocks[2], stride=strides[2])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # (N, 1, L) --> (N, 4, L)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


def get_lout(lin, strides):
    import math
    lout = lin
    for stride in strides:
        lout = math.floor(float(lout-1) / stride + 1)
    return lout


def ResNet3(out_channels=128, strides=(1, 2, 2), init_channels=1, in_planes=4):
    """ResNet with 3 blocks"""
    return ResNet_3layers(BasicBlock, [1, 1, 1], strides, out_channels, init_channels, in_planes)


# model ===============================================
class ModelBiLSTM(nn.Module):
    """
    BiLSTM network process sequence/signal features
    """
    def __init__(self, seq_len=13, signal_len=16, num_layers1=3, num_layers2=1, num_classes=2,
                 dropout_rate=0.5, hidden_size=256,
                 vocab_size=16, embedding_size=4, is_base=True, is_signallen=True,
                 module="both_bilstm"):
        super(ModelBiLSTM, self).__init__()
        self.model_type = 'BiLSTM'
        self.module = module

        self.seq_len = seq_len
        self.signal_len = signal_len
        self.num_layers1 = num_layers1  # for combined (seq+signal) feature
        self.num_layers2 = num_layers2  # for seq and signal feature separately
        self.num_classes = num_classes

        self.hidden_size = hidden_size

        if self.module == "both_bilstm":
            self.nhid_seq = self.hidden_size // 2
            self.nhid_signal = self.hidden_size - self.nhid_seq
        elif self.module == "seq_bilstm":
            self.nhid_seq = self.hidden_size
        elif self.module == "signal_bilstm":
            self.nhid_signal = self.hidden_size
        else:
            raise ValueError("--model_type is not right!")

        # seq feature
        if self.module != "signal_bilstm":
            self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base
            self.is_base = is_base
            self.is_signallen = is_signallen
            self.sigfea_num = 3 if self.is_signallen else 2
            if is_base:
                self.lstm_seq = nn.LSTM(embedding_size + self.sigfea_num, self.nhid_seq, self.num_layers2,
                                        dropout=dropout_rate, batch_first=True, bidirectional=True)
            else:
                self.lstm_seq = nn.LSTM(self.sigfea_num, self.nhid_seq, self.num_layers2,
                                        dropout=dropout_rate, batch_first=True, bidirectional=True)
            self.fc_seq = nn.Linear(self.nhid_seq * 2, self.nhid_seq)
            # self.dropout_seq = nn.Dropout(p=dropout_rate)
            self.relu_seq = nn.ReLU()

        # signal feature
        if self.module != "seq_bilstm":
            # self.convs = ResNet3(self.nhid_signal, (1, 1, 1), self.signal_len, self.signal_len)  # (N, C, L)
            self.lstm_signal = nn.LSTM(self.signal_len, self.nhid_signal, self.num_layers2,
                                       dropout=dropout_rate, batch_first=True, bidirectional=True)
            self.fc_signal = nn.Linear(self.nhid_signal * 2, self.nhid_signal)
            # self.dropout_signal = nn.Dropout(p=dropout_rate)
            self.relu_signal = nn.ReLU()

        # combined
        self.lstm_comb = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers1,
                                 dropout=dropout_rate, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # 2 for bidirection
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def get_model_type(self):
        return self.model_type

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        c0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        if use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return h0, c0

    def forward(self, kmer, base_means, base_stds, base_signal_lens, signals):
        # seq feature ============================================
        # kmer, base_means, base_stds, base_signal_lens
        if self.module != "signal_bilstm":
            base_means = torch.reshape(base_means, (-1, self.seq_len, 1)).float()
            base_stds = torch.reshape(base_stds, (-1, self.seq_len, 1)).float()
            base_signal_lens = torch.reshape(base_signal_lens, (-1, self.seq_len, 1)).float()
            if self.is_base:
                kmer_embed = self.embed(kmer.long())
                if self.is_signallen:
                    out_seq = torch.cat((kmer_embed, base_means, base_stds, base_signal_lens), 2)  # (N, L, C)
                else:
                    out_seq = torch.cat((kmer_embed, base_means, base_stds), 2)  # (N, L, C)
            else:
                if self.is_signallen:
                    out_seq = torch.cat((base_means, base_stds, base_signal_lens), 2)  # (N, L, C)
                else:
                    out_seq = torch.cat((base_means, base_stds), 2)  # (N, L, C)
            out_seq, _ = self.lstm_seq(out_seq, self.init_hidden(out_seq.size(0),
                                                                 self.num_layers2,
                                                                 self.nhid_seq))  # (N, L, nhid_seq*2)
            out_seq = self.fc_seq(out_seq)  # (N, L, nhid_seq)
            # out_seq = self.dropout_seq(out_seq)
            out_seq = self.relu_seq(out_seq)

        # signal feature ==========================================
        # sgianls (L*C)
        if self.module != "seq_bilstm":
            out_signal = signals.float()
            # resnet ---
            # out_signal = out_signal.transpose(1, 2)  # (N, C, L)
            # out_signal = self.convs(out_signal)  # (N, nhid_signal, L)
            # out_signal = out_signal.transpose(1, 2)  # (N, L, nhid_signal)
            # lstm ---
            out_signal, _ = self.lstm_signal(out_signal, self.init_hidden(out_signal.size(0),
                                                                          self.num_layers2,
                                                                          self.nhid_signal))
            out_signal = self.fc_signal(out_signal)  # (N, L, nhid_signal)
            # out_signal = self.dropout_signal(out_signal)
            out_signal = self.relu_signal(out_signal)

        # combined ================================================
        if self.module == "seq_bilstm":
            out = out_seq
        elif self.module == "signal_bilstm":
            out = out_signal
        elif self.module == "both_bilstm":
            out = torch.cat((out_seq, out_signal), 2)  # (N, L, hidden_size)
        out, _ = self.lstm_comb(out, self.init_hidden(out.size(0),
                                                      self.num_layers1,
                                                      self.hidden_size))  # (N, L, hidden_size*2)
        out_fwd_last = out[:, -1, :self.hidden_size]
        out_bwd_last = out[:, 0, self.hidden_size:]
        out = torch.cat((out_fwd_last, out_bwd_last), 1)

        # decode
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.dropout2(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out, self.softmax(out)
