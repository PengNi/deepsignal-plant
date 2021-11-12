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

from .models import ModelBiLSTM
from .dataloader import SignalFeaData2
from .dataloader import clear_linecache
from .utils.process_utils import display_args
from .utils.process_utils import str2bool

from .utils.constants_torch import use_cuda


def train(args):
    """

    :param args: train_sample_file, valid_sample_file, hyperparameters
    :return: directory contains model_params_checkpoint_file
    """
    total_start = time.time()
    # torch.manual_seed(args.seed)

    print("[train] start..")
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!")

    print("reading data..")
    train_dataset = SignalFeaData2(args.train_file)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    valid_dataset = SignalFeaData2(args.valid_file)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False)

    model_dir = args.model_dir
    model_regex = re.compile(r"" + args.model_type + "\.b\d+_s\d+_epoch\d+\.ckpt*")
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile):
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    model = ModelBiLSTM(args.seq_len, args.signal_len, args.layernum1, args.layernum2, args.class_num,
                        args.dropout_rate, args.hid_rnn,
                        args.n_vocab, args.n_embed, str2bool(args.is_base), str2bool(args.is_signallen),
                        args.model_type)
    if use_cuda:
        model = model.cuda()
    if args.init_model is not None:
        print("loading pre-trained model: {}".format(args.init_model))
        para_dict = torch.load(args.init_model) if use_cuda else torch.load(args.init_model,
                                                                            map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        model_dict.update(para_dict)
        model.load_state_dict(model_dict)

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_rank)
    if args.optim_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim_type == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)
    elif args.optim_type == "Ranger":
        # use Ranger optimizer
        # refer to https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
        # needs python>=3.6
        try:
            from .utils.ranger2020 import Ranger
        except ImportError:
            raise ImportError("please check if ranger2020.py is in utils/ dir!")
        optimizer = Ranger(model.parameters(), lr=args.lr)
    else:
        raise ValueError("optim_type is not right!")
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)

    # Train the model
    total_step = len(train_loader)
    print("total_step: {}".format(total_step))
    curr_best_accuracy = 0
    model.train()
    # train at most max_epoch_num epochs
    for epoch in range(args.max_epoch_num):
        curr_best_accuracy_epoch = 0
        tlosses = []
        start = time.time()
        for i, sfeatures in enumerate(train_loader):
            _, kmer, base_means, base_stds, base_signal_lens, signals, labels = sfeatures
            if use_cuda:
                kmer = kmer.cuda()
                base_means = base_means.cuda()
                base_stds = base_stds.cuda()
                base_signal_lens = base_signal_lens.cuda()
                signals = signals.cuda()
                labels = labels.cuda()

            # Forward pass
            outputs, logits = model(kmer, base_means, base_stds, base_signal_lens, signals)
            loss = criterion(outputs, labels)
            tlosses.append(loss.detach().item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i + 1) % args.step_interval == 0:
                model.eval()
                with torch.no_grad():
                    vlosses, vlabels_total, vpredicted_total = [], [], []
                    for vi, vsfeatures in enumerate(valid_loader):
                        _, vkmer, vbase_means, vbase_stds, vbase_signal_lens, vsignals, vlabels = vsfeatures
                        if use_cuda:
                            vkmer = vkmer.cuda()
                            vbase_means = vbase_means.cuda()
                            vbase_stds = vbase_stds.cuda()
                            vbase_signal_lens = vbase_signal_lens.cuda()
                            vsignals = vsignals.cuda()
                            vlabels = vlabels.cuda()
                        voutputs, vlogits = model(vkmer, vbase_means, vbase_stds, vbase_signal_lens, vsignals)
                        vloss = criterion(voutputs, vlabels)

                        _, vpredicted = torch.max(vlogits.data, 1)

                        if use_cuda:
                            vlabels = vlabels.cpu()
                            vpredicted = vpredicted.cpu()
                        # i_accuracy = metrics.accuracy_score(vlabels.numpy(), vpredicted)
                        # i_precision = metrics.precision_score(vlabels.numpy(), vpredicted)
                        # i_recall = metrics.recall_score(vlabels.numpy(), vpredicted)

                        # vaccus.append(i_accuracy)
                        # vprecs.append(i_precision)
                        # vrecas.append(i_recall)
                        vlosses.append(vloss.item())
                        vlabels_total += vlabels
                        vpredicted_total += vpredicted

                    v_accuracy = metrics.accuracy_score(vlabels_total, vpredicted_total)
                    v_precision = metrics.precision_score(vlabels_total, vpredicted_total)
                    v_recall = metrics.recall_score(vlabels_total, vpredicted_total)
                    if v_accuracy > curr_best_accuracy_epoch:
                        curr_best_accuracy_epoch = v_accuracy
                        if curr_best_accuracy_epoch > curr_best_accuracy - 0.0005:
                            torch.save(model.state_dict(),
                                       model_dir + args.model_type + '.b{}_s{}_epoch{}.ckpt'.format(args.seq_len,
                                                                                                    args.signal_len,
                                                                                                    epoch + 1))

                    time_cost = time.time() - start
                    print('Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; '
                          'ValidLoss: {:.4f}, '
                          'Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
                          'curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s'
                          .format(epoch + 1, args.max_epoch_num, i + 1, total_step, np.mean(tlosses),
                                  np.mean(vlosses), v_accuracy, v_precision, v_recall,
                                  curr_best_accuracy_epoch, time_cost))
                    tlosses = []
                    start = time.time()
                    sys.stdout.flush()
                model.train()
        scheduler.step()
        if curr_best_accuracy_epoch > curr_best_accuracy:
            curr_best_accuracy = curr_best_accuracy_epoch
        else:
            if epoch >= args.min_epoch_num - 1:
                print("best accuracy: {}, early stop!".format(curr_best_accuracy))
                break

    endtime = time.time()
    clear_linecache()
    print("[train] training cost {} seconds".format(endtime - total_start))


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)

    # model input
    parser.add_argument('--model_type', type=str, default="both_bilstm",
                        choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
                        required=False,
                        help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
                             "'both_bilstm' means to use both seq and signal bilstm, default: both_bilstm")
    parser.add_argument('--seq_len', type=int, default=13, required=False,
                        help="len of kmer. default 13")
    parser.add_argument('--signal_len', type=int, default=16, required=False,
                        help="the number of signals of one base to be used in deepsignal_plant, default 16")

    # model param
    parser.add_argument('--layernum1', type=int, default=3,
                        required=False, help="lstm layer num for combined feature, default 3")
    parser.add_argument('--layernum2', type=int, default=1,
                        required=False, help="lstm layer num for seq feature (and for signal feature too), default 1")
    parser.add_argument('--class_num', type=int, default=2, required=False)
    parser.add_argument('--dropout_rate', type=float, default=0.5, required=False)
    parser.add_argument('--n_vocab', type=int, default=16, required=False,
                        help="base_seq vocab_size (15 base kinds from iupac)")
    parser.add_argument('--n_embed', type=int, default=4, required=False,
                        help="base_seq embedding_size")
    parser.add_argument('--is_base', type=str, default="yes", required=False,
                        help="is using base features in seq model, default yes")
    parser.add_argument('--is_signallen', type=str, default="yes", required=False,
                        help="is using signal length feature of each base in seq model, default yes")

    # BiLSTM model param
    parser.add_argument('--hid_rnn', type=int, default=256, required=False,
                        help="BiLSTM hidden_size for combined feature")

    # model training
    parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "RMSprop", "SGD",
                                                                           "Ranger"],
                        required=False, help="type of optimizer to use, 'Adam' or 'SGD' or 'RMSprop' or 'Ranger', "
                                             "default Adam")
    parser.add_argument('--batch_size', type=int, default=512, required=False)
    parser.add_argument('--lr', type=float, default=0.001, required=False)
    parser.add_argument('--lr_decay', type=float, default=0.1, required=False)
    parser.add_argument('--lr_decay_step', type=int, default=2, required=False)
    parser.add_argument("--max_epoch_num", action="store", default=10, type=int,
                        required=False, help="max epoch num, default 10")
    parser.add_argument("--min_epoch_num", action="store", default=5, type=int,
                        required=False, help="min epoch num, default 5")
    parser.add_argument('--step_interval', type=int, default=100, required=False)

    parser.add_argument('--pos_weight', type=float, default=1.0, required=False)
    parser.add_argument('--init_model', type=str, default=None, required=False,
                        help="file path of pre-trained model parameters to load before training")
    # parser.add_argument('--seed', type=int, default=1234,
    #                     help='random seed')

    # else
    parser.add_argument('--tmpdir', type=str, default="/tmp", required=False)

    args = parser.parse_args()

    print("[main] start..")
    total_start = time.time()

    display_args(args)

    train(args)

    endtime = time.time()
    print("[main] costs {} seconds".format(endtime - total_start))


if __name__ == '__main__':
    main()
