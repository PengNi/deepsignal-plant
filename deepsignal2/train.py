# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from sklearn import metrics
import numpy as np
import argparse
import os
import sys
import time
import re

from modules import SignalEncoderClassifier
from modules import SeqEncoderClassifier
from dataloader import SignalFeaData
from utils.process_utils import str2bool
from utils.process_utils import display_args

from utils.constants_torch import use_cuda


def train_signal(args):
    total_start = time.time()

    print("[train]start..")
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!")

    print("reading data..")
    train_dataset = SignalFeaData(args.train_file)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    valid_dataset = SignalFeaData(args.valid_file)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    model_dir = args.model_dir
    model_regex = re.compile(r"epoch\d+\.pkl*")
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile):
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    model = SignalEncoderClassifier(args.signal_len, args.d_model, args.d_ff, args.n_head,
                                    args.layer_num, args.class_num, args.dropout_rate)
    if use_cuda:
        model = model.cuda()

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    total_step = len(train_loader)
    print("total_step: {}".format(total_step))
    start = time.time()
    curr_best_accuracy = 0
    for epoch in range(args.epoch_num):
        curr_best_accuracy_epoch = 0
        for i, sfeatures in enumerate(train_loader):
            _, _, _, _, _, cent_signals, labels = sfeatures
            if use_cuda:
                cent_signals = cent_signals.cuda()
                labels = labels.cuda()

            model.train()

            # Forward pass
            outputs, _ = model(cent_signals)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % args.step_interval == 0:
                model.eval()
                vlosses, vaccus, vprecs, vrecas = [], [], [], []
                for vi, vsfeatures in enumerate(valid_loader):
                    _, _, _, _, _, vcent_signals, vlabels = vsfeatures
                    if use_cuda:
                        vcent_signals = vcent_signals.cuda()
                        vlabels = vlabels.cuda()

                    voutputs, _ = model(vcent_signals)
                    vloss = criterion(voutputs, vlabels)

                    _, vpredicted = torch.max(voutputs.data, 1)

                    if use_cuda:
                        vlabels = vlabels.cpu()
                        vpredicted = vpredicted.cpu()
                    i_accuracy = metrics.accuracy_score(vlabels.numpy(), vpredicted)
                    i_precision = metrics.precision_score(vlabels.numpy(), vpredicted)
                    i_recall = metrics.recall_score(vlabels.numpy(), vpredicted)

                    vaccus.append(i_accuracy)
                    vprecs.append(i_precision)
                    vrecas.append(i_recall)
                    vlosses.append(vloss.item())

                if np.mean(vaccus) > curr_best_accuracy_epoch:
                    torch.save(model.state_dict(), model_dir + 'epoch{}.pkl'.format(epoch))
                    curr_best_accuracy_epoch = np.mean(vaccus)

                time_cost = time.time() - start
                print('Epoch [{}/{}], Step [{}/{}], ValidSet Loss: {:.4f}, '
                      'Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
                      'curr_epoch_best_accuracy: {:.4f}, Time: {:.2f}s'
                      .format(epoch + 1, args.epoch_num, i + 1, total_step, np.mean(vlosses),
                              np.mean(vaccus), np.mean(vprecs), np.mean(vrecas),
                              curr_best_accuracy_epoch, time_cost))
                start = time.time()
                sys.stdout.flush()
        if curr_best_accuracy_epoch > curr_best_accuracy:
            curr_best_accuracy = curr_best_accuracy_epoch
        else:
            print("best accuracy: {}, early stop!".format(curr_best_accuracy))
            break

    endtime = time.time()
    print("[train]training cost {} seconds".format(endtime - total_start))


def train_sequence(args):
    total_start = time.time()

    print("[train]start..")
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!")

    print("reading data..")
    train_dataset = SignalFeaData(args.train_file)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    valid_dataset = SignalFeaData(args.valid_file)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    model_dir = args.model_dir
    model_regex = re.compile(r"epoch\d+\.pkl*")
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile):
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    model = SeqEncoderClassifier(args.seq_len, args.d_model, args.d_ff, args.n_head,
                                 args.layer_num, args.class_num, args.dropout_rate)
    if use_cuda:
        model = model.cuda()

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    total_step = len(train_loader)
    print("total_step: {}".format(total_step))
    start = time.time()
    curr_best_accuracy = 0
    for epoch in range(args.epoch_num):
        curr_best_accuracy_epoch = 0
        for i, sfeatures in enumerate(train_loader):
            _, kmer, base_means, base_stds, base_signal_lens, _, labels = sfeatures
            if use_cuda:
                kmer = kmer.cuda()
                base_means = base_means.cuda()
                base_stds = base_stds.cuda()
                base_signal_lens = base_signal_lens.cuda()
                labels = labels.cuda()

            model.train()

            # Forward pass
            outputs, _ = model(kmer, base_means, base_stds, base_signal_lens)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % args.step_interval == 0:
                model.eval()
                vlosses, vaccus, vprecs, vrecas = [], [], [], []
                for vi, vsfeatures in enumerate(valid_loader):
                    _, vkmer, vbase_means, vbase_stds, vbase_signal_lens, _, vlabels = vsfeatures
                    if use_cuda:
                        vkmer = vkmer.cuda()
                        vbase_means = vbase_means.cuda()
                        vbase_stds = vbase_stds.cuda()
                        vbase_signal_lens = vbase_signal_lens.cuda()
                        vlabels = vlabels.cuda()

                    voutputs, _ = model(vkmer, vbase_means, vbase_stds, vbase_signal_lens)
                    vloss = criterion(voutputs, vlabels)

                    _, vpredicted = torch.max(voutputs.data, 1)

                    if use_cuda:
                        vlabels = vlabels.cpu()
                        vpredicted = vpredicted.cpu()
                    i_accuracy = metrics.accuracy_score(vlabels.numpy(), vpredicted)
                    i_precision = metrics.precision_score(vlabels.numpy(), vpredicted)
                    i_recall = metrics.recall_score(vlabels.numpy(), vpredicted)

                    vaccus.append(i_accuracy)
                    vprecs.append(i_precision)
                    vrecas.append(i_recall)
                    vlosses.append(vloss.item())

                if np.mean(vaccus) > curr_best_accuracy_epoch:
                    curr_best_accuracy_epoch = np.mean(vaccus)
                    if curr_best_accuracy_epoch > curr_best_accuracy:
                        torch.save(model.state_dict(), model_dir + 'epoch{}.pkl'.format(epoch))

                time_cost = time.time() - start
                print('Epoch [{}/{}], Step [{}/{}], ValidSet Loss: {:.4f}, '
                      'Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
                      'curr_epoch_best_accuracy: {:.4f}, Time: {:.2f}s'
                      .format(epoch + 1, args.epoch_num, i + 1, total_step, np.mean(vlosses),
                              np.mean(vaccus), np.mean(vprecs), np.mean(vrecas),
                              curr_best_accuracy_epoch, time_cost))
                start = time.time()
                sys.stdout.flush()
        if curr_best_accuracy_epoch > curr_best_accuracy:
            curr_best_accuracy = curr_best_accuracy
        else:
            print("best accuracy: {}, early stop!".format(curr_best_accuracy))
            break

    endtime = time.time()
    print("[train]training cost {} seconds".format(endtime - total_start))


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)

    # model param
    parser.add_argument('--is_seq_module', type=str, default='yes', required=False,
                        help="use seq_module or not, default yes. no means using signal_module")
    parser.add_argument('--seq_len', type=int, default=11, required=False)
    parser.add_argument('--signal_len', type=int, default=128, required=False)
    parser.add_argument('--layer_num', type=int, default=3,
                        required=False, help="encoder layer num")
    parser.add_argument('--class_num', type=int, default=2, required=False)
    parser.add_argument('--dropout_rate', type=float, default=0.1, required=False)
    parser.add_argument('--d_model', type=int, default=512, required=False)
    parser.add_argument('--d_ff', type=int, default=1024, required=False)
    parser.add_argument('--n_head', type=int, default=4, required=False)

    # model training
    parser.add_argument('--batch_size', type=int, default=512, required=False)
    parser.add_argument('--lr', type=float, default=0.001, required=False)
    parser.add_argument('--epoch_num', type=int, default=10, required=False)
    parser.add_argument('--step_interval', type=int, default=100, required=False)

    parser.add_argument('--pos_weight', type=float, default=1.0, required=False)

    # else
    parser.add_argument('--tmpdir', type=str, default="/tmp", required=False)

    args = parser.parse_args()

    print("[main] start..")
    total_start = time.time()

    display_args(args)

    if str2bool(args.is_seq_module):
        train_sequence(args)
    else:
        train_signal(args)

    endtime = time.time()
    print("[main] costs {} seconds".format(endtime - total_start))


if __name__ == '__main__':
    main()
