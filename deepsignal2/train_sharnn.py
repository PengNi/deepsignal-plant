# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
import numpy as np
import argparse
import os
import sys
import time
import re

from models import SHARNN
from dataloader import SignalFeaData
from utils.process_utils import display_args
from utils.process_utils import str2bool

from utils.constants_torch import use_cuda


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is None:
        return None
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(args):
    total_start = time.time()
    torch.manual_seed(args.seed)

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
    model_regex = re.compile(r"epoch\d+\.ckpt*")
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile):
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    model = SHARNN(args.seq_len, args.emsize, args.nhid, args.nlayers, args.class_num,
                   args.dropout_rate, args.dropout_rate, args.dropout_rate, args.dropout_rate,
                   args.wdrop, args.tied)
    if use_cuda:
        model = model.cuda()

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    # Train the model
    total_step = len(train_loader)
    print("total_step: {}".format(total_step))
    start = time.time()
    curr_best_accuracy = 0
    for epoch in range(args.max_epoch_num):
        curr_best_accuracy_epoch = 0
        h_hid, h_mems = None, None
        for i, sfeatures in enumerate(train_loader):
            if len(sfeatures[0]) < args.batch_size:
                continue
            _, kmer, base_means, base_stds, base_signal_lens, _, labels = sfeatures
            if use_cuda:
                kmer = kmer.cuda()
                base_means = base_means.cuda()
                base_stds = base_stds.cuda()
                base_signal_lens = base_signal_lens.cuda()
                # signals = signals.cuda()
                labels = labels.cuda()

            model.train()

            # Forward pass
            if str2bool(args.prehid):
                h, hlogits, h_hid, h_mems = model(kmer, base_means, base_stds, base_signal_lens,
                                                  hidden=h_hid, mems=h_mems)
                if h_hid is not None:
                    h_hid = repackage_hidden(h_hid)
                if h_mems is not None:
                    h_mems = repackage_hidden(h_mems)
            else:
                h, hlogits, _, _ = model(kmer, base_means, base_stds, base_signal_lens,
                                         hidden=h_hid, mems=h_mems)

            loss = criterion(h, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % args.step_interval == 0:
                model.eval()
                v_hid, v_mems = None, None
                vlosses, vaccus, vprecs, vrecas = [], [], [], []
                for vi, vsfeatures in enumerate(valid_loader):
                    if len(vsfeatures[0]) < args.batch_size:
                        continue
                    _, vkmer, vbase_means, vbase_stds, vbase_signal_lens, _, vlabels = vsfeatures
                    if use_cuda:
                        vkmer = vkmer.cuda()
                        vbase_means = vbase_means.cuda()
                        vbase_stds = vbase_stds.cuda()
                        vbase_signal_lens = vbase_signal_lens.cuda()
                        # vsignals = vsignals.cuda()
                        vlabels = vlabels.cuda()

                    if str2bool(args.prehid):
                        v, vlogits, v_hid, v_mems = model(vkmer, vbase_means, vbase_stds, vbase_signal_lens,
                                                          hidden=v_hid, mems=v_mems)
                        if v_hid is not None:
                            v_hid = repackage_hidden(v_hid)
                        if v_mems is not None:
                            v_mems = repackage_hidden(v_mems)
                    else:
                        v, vlogits, _, _ = model(vkmer, vbase_means, vbase_stds, vbase_signal_lens,
                                                 hidden=v_hid, mems=v_mems)
                    vloss = criterion(v, vlabels)

                    _, vpredicted = torch.max(vlogits.data, 1)

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
                    if curr_best_accuracy_epoch > curr_best_accuracy - 0.001:
                        torch.save(model.state_dict(), model_dir + 'epoch{}.ckpt'.format(epoch))

                time_cost = time.time() - start
                print('Epoch [{}/{}], Step [{}/{}], ValidSet Loss: {:.4f}, '
                      'Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
                      'curr_epoch_best_accuracy: {:.4f}, Time: {:.2f}s'
                      .format(epoch + 1, args.max_epoch_num, i + 1, total_step, np.mean(vlosses),
                              np.mean(vaccus), np.mean(vprecs), np.mean(vrecas),
                              curr_best_accuracy_epoch, time_cost))
                start = time.time()
                sys.stdout.flush()
        scheduler.step()
        if curr_best_accuracy_epoch > curr_best_accuracy:
            curr_best_accuracy = curr_best_accuracy_epoch
        else:
            if epoch >= args.min_epoch_num - 1:
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
    parser.add_argument('--seq_len', type=int, default=11, required=False)
    parser.add_argument('--signal_len', type=int, default=128, required=False)
    parser.add_argument('--layer_num', type=int, default=3,
                        required=False, help="encoder layer num")
    parser.add_argument('--class_num', type=int, default=2, required=False)
    parser.add_argument('--dropout_rate', type=float, default=0.3, required=False)
    parser.add_argument('--emsize', type=int, default=256,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=512,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--wdrop', type=float, default=0.0,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--prehid', type=str, default="no", required=False,
                        help="")

    # model training
    parser.add_argument('--batch_size', type=int, default=512, required=False)
    parser.add_argument('--lr', type=float, default=0.001, required=False)
    parser.add_argument("--max_epoch_num", action="store", default=10, type=int,
                        required=False, help="max epoch num, default 10")
    parser.add_argument("--min_epoch_num", action="store", default=5, type=int,
                        required=False, help="min epoch num, default 5")
    parser.add_argument('--step_interval', type=int, default=100, required=False)

    parser.add_argument('--pos_weight', type=float, default=1.0, required=False)
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')

    # else
    parser.add_argument('--tmpdir', type=str, default="/tmp", required=False)

    args = parser.parse_args()
    args.tied = True

    print("[main] start..")
    total_start = time.time()

    display_args(args)

    train(args)

    endtime = time.time()
    print("[main] costs {} seconds".format(endtime - total_start))


if __name__ == '__main__':
    main()
