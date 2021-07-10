from __future__ import absolute_import

import argparse
import time
import os
import sys
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from .models import ModelBiLSTM
from .dataloader import SignalFeaData2
from .dataloader import clear_linecache

from .utils.constants_torch import use_cuda
from .utils.process_utils import str2bool
from .utils.process_utils import random_select_file_rows_s
# from utils.process_utils import random_select_file_rows
from .utils.process_utils import count_line_num
from .utils.process_utils import concat_two_files

from .utils.process_utils import select_negsamples_asposkmer
from .utils.process_utils import get_model_type_str


def train_1time(train_file, valid_file, valid_lidxs, args):
    """
    use train_file to train model, then score the probs of the samples in valid_file
    :param train_file:
    :param valid_file:
    :param valid_lidxs:
    :param args:
    :return:
    """
    # ===========
    train_dataset = SignalFeaData2(train_file)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    model = ModelBiLSTM(args.seq_len, args.signal_len, args.layernum1, args.layernum2, args.class_num,
                        args.dropout_rate, args.hid_rnn,
                        args.n_vocab, args.n_embed, str2bool(args.is_base), str2bool(args.is_signallen),
                        args.model_type)
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
    print("train total_step: {}".format(total_step))
    start = time.time()
    model.train()
    for epoch in range(args.epoch_num):
        test_accus = []
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
            outputs, tlogits = model(kmer, base_means, base_stds, base_signal_lens, signals)

            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i + 1) % args.step_interval == 0:
                _, tpredicted = torch.max(tlogits.data, 1)

                tlabels = labels
                if use_cuda:
                    tlabels = labels.cpu()
                    tpredicted = tpredicted.cpu()
                i_accuracy = metrics.accuracy_score(tlabels.numpy(), tpredicted)
                i_precision = metrics.precision_score(tlabels.numpy(), tpredicted)
                i_recall = metrics.recall_score(tlabels.numpy(), tpredicted)

                test_accus.append(i_accuracy)

                endtime = time.time()
                print('Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}, '
                      'Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
                      'Time: {:.2f}s'
                      .format(epoch + 1, args.epoch_num, i + 1, total_step, loss.item(),
                              i_accuracy, i_precision, i_recall, endtime - start))
                sys.stdout.flush()
                start = time.time()
        scheduler.step()
        if np.mean(test_accus) >= 0.95:
            break

    # valid data
    valid_dataset = SignalFeaData2(valid_file)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False)
    total_step = len(valid_loader)
    print("valid total_step: {}".format(total_step))
    model.eval()
    vlosses, vaccus, vprecs, vrecas = [], [], [], []
    lineidx_cnt = 0
    idx2aclogits = {}
    start = time.time()
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
            vlogits = vlogits.cpu()

        for alogit in vlogits.detach().numpy():
            idx2aclogits[valid_lidxs[lineidx_cnt]] = alogit[1]
            lineidx_cnt += 1

        i_accuracy = metrics.accuracy_score(vlabels.numpy(), vpredicted)
        i_precision = metrics.precision_score(vlabels.numpy(), vpredicted)
        i_recall = metrics.recall_score(vlabels.numpy(), vpredicted)

        vaccus.append(i_accuracy)
        vprecs.append(i_precision)
        vrecas.append(i_recall)

        if (vi + 1) % args.step_interval == 0:
            endtime = time.time()
            print('===Test, Step [{}/{}], ValidLoss: {:.4f}, '
                  'Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
                  'Time: {:.2f}s'
                  .format(vi+1, total_step, vloss.item(), i_accuracy, i_precision, i_recall, endtime - start))
            sys.stdout.flush()
            start = time.time()

    print("===Test, Total Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(np.mean(vaccus),
                                                                                      np.mean(vprecs),
                                                                                      np.mean(vrecas)))
    del model
    # treat linecache carefully
    clear_linecache()
    return idx2aclogits


def train_rounds(train_file, iterstr, args, modeltype_str):
    """
    repeat rounds of splitting train_file to train_then_valid
    :param train_file:
    :param iterstr:
    :param args:
    :param modeltype_str:
    :return:
    """
    print("\n##########Train Cross Rank##########")
    total_num = count_line_num(train_file, False)
    half_num = total_num // 2
    fname, fext = os.path.splitext(train_file)
    idxs2logtis_all = {}
    for i in range(0, total_num):
        idxs2logtis_all[i] = []

    for i in range(0, args.rounds):
        print("##########Train Cross Rank, Iter {}, Round {}##########".format(iterstr, i+1))
        if train_file == args.train_file:
            train_file1 = fname + "." + modeltype_str + ".half1" + fext
            train_file2 = fname + "." + modeltype_str + ".half2" + fext
        else:
            train_file1 = fname + ".half1" + fext
            train_file2 = fname + ".half2" + fext
        lidxs1, lidxs2 = random_select_file_rows_s(train_file, train_file1, train_file2,
                                                   half_num, False)
        print("##########Train Cross Rank, Iter {}, Round {}, part1##########".format(iterstr, i + 1))
        idxs22logits = train_1time(train_file1, train_file2, lidxs2, args)
        print("##########Train Cross Rank, Iter {}, Round {}, part2##########".format(iterstr, i + 1))
        idxs12logits = train_1time(train_file2, train_file1, lidxs1, args)
        for idx in idxs22logits.keys():
            idxs2logtis_all[idx].append(idxs22logits[idx])
        for idx in idxs12logits.keys():
            idxs2logtis_all[idx].append(idxs12logits[idx])

        os.remove(train_file1)
        os.remove(train_file2)
    print("##########Train Cross Rank, finished!##########")
    sys.stdout.flush()
    return idxs2logtis_all


def clean_samples(train_file, idx2logits, score_cf, is_filter_fn, ori_train_file, modeltype_str):
    # clean train samples ===
    print("\n###### clean the samples ######")
    idx2probs = dict()
    for idx in idx2logits.keys():
        probs = idx2logits[idx]
        meanprob = np.mean(probs)
        stdprob = np.std(probs)
        idx2probs[idx] = [meanprob, stdprob]

    idx2prob_pos, idx2prob_neg = [], []
    with open(train_file, 'r') as rf:
        linecnt = 0
        for line in rf:
            words = line.strip().split("\t")
            label = int(words[-1])
            if label == 1:
                idx2prob_pos.append((linecnt, idx2probs[linecnt][0], idx2probs[linecnt][1]))
            else:
                idx2prob_neg.append((linecnt, idx2probs[linecnt][0], idx2probs[linecnt][1]))
            linecnt += 1

    print("There are {} positive, {} negative samples in total;".format(len(idx2prob_pos),
                                                                        len(idx2prob_neg)))

    pos_hc, neg_hc = set(), set()

    idx2prob_pos = sorted(idx2prob_pos, key=lambda x: x[1], reverse=True)
    for idx2prob in idx2prob_pos:
        if idx2prob[1] >= score_cf:
            pos_hc.add(idx2prob[0])
    if is_filter_fn:
        idx2prob_neg = sorted(idx2prob_neg, key=lambda x: x[1])
        for idx2prob in idx2prob_neg:
            if idx2prob[1] < 1 - score_cf:
                neg_hc.add(idx2prob[0])

    left_ratio = float(len(pos_hc)) / len(idx2prob_pos) if len(idx2prob_pos) > 0 else 0
    left_ratio2 = float(len(neg_hc)) / len(idx2prob_neg) if len(idx2prob_neg) > 0 else 0
    print("{} ({}) high quality positive samples left, "
          "{} ({}) high quality negative samples left".format(len(pos_hc),
                                                              round(left_ratio, 6),
                                                              len(neg_hc),
                                                              round(left_ratio2, 6)))

    # re-write train set
    fname, fext = os.path.splitext(train_file)
    if train_file == ori_train_file:
        train_clean_pos_file = fname + "." + modeltype_str + ".pos.cf" + str(score_cf) + fext
    else:
        train_clean_pos_file = fname + ".pos.cf" + str(score_cf) + fext
    wfp = open(train_clean_pos_file, 'w')
    if is_filter_fn:
        if train_file == ori_train_file:
            train_clean_neg_file = fname + "." + modeltype_str + ".neg.cf" + str(score_cf) + fext
        else:
            train_clean_neg_file = fname + ".neg.cf" + str(score_cf) + fext
        wfn = open(train_clean_neg_file, 'w')
    lidx = 0
    with open(train_file, 'r') as rf:
        for line in rf:
            if lidx in pos_hc:
                wfp.write(line)
            elif is_filter_fn and lidx in neg_hc:
                wfn.write(line)
            lidx += 1
    wfp.close()
    if is_filter_fn:
        wfn.close()

    print("###### clean the samples, finished! ######")
    sys.stdout.flush()

    if is_filter_fn:
        left_ratio = (left_ratio + left_ratio2) / 2
        return train_clean_pos_file, left_ratio, train_clean_neg_file
    else:
        return train_clean_pos_file, left_ratio, None


def _get_all_negative_samples(train_file, modeltype_str):
    fname, fext = os.path.splitext(train_file)
    train_neg_file = fname + ".neg_all" + "." + modeltype_str + fext

    wf = open(train_neg_file, "w")
    with open(train_file, 'r') as rf:
        for line in rf:
            words = line.strip().split("\t")
            label = int(words[-1])
            if label == 0:
                wf.write(line)
    wf.close()
    return train_neg_file


def _output_linenumber2probs(wfile, idx2logits):
    wf = open(wfile, "w")
    for idx in sorted(list(idx2logits.keys())):
        wf.write("\t".join([str(idx), str(np.mean(idx2logits[idx]))]) + "\n")
    wf.close()


def denoise(args):
    total_start = time.time()

    iterations = args.iterations

    train_file = args.train_file
    modeltype_str = get_model_type_str(args.model_type, str2bool(args.is_base), str2bool(args.is_signallen))

    # filter neg samples ===
    train_neg_file = _get_all_negative_samples(train_file, modeltype_str)

    for iter_c in range(iterations):
        print("\n###### cross rank to clean samples, Iter: {} ######".format(iter_c + 1))
        # cross rank
        iterstr = str(iter_c + 1)
        idxs2logtis_all = train_rounds(train_file, iterstr, args, modeltype_str)

        # output probs of 1 iteration
        if iter_c == 0 and args.fst_iter_prob:
            wfile = train_file + ".probs_1stiter.txt"
            _output_linenumber2probs(wfile, idxs2logtis_all)

        is_filter_fn = str2bool(args.is_filter_fn)
        train_clean_pos_file, left_ratio, train_clean_neg_file = clean_samples(train_file, idxs2logtis_all,
                                                                               args.score_cf, is_filter_fn,
                                                                               args.train_file, modeltype_str)
        if train_file != args.train_file:
            os.remove(train_file)

        # concat new train_file
        print("\n#####concat denoied file#####")
        pos_num = count_line_num(train_clean_pos_file)
        if pos_num > 0:
            fname, fext = os.path.splitext(train_neg_file)
            train_seled_neg_file = fname + ".r" + str(pos_num) + fext
            if train_clean_neg_file is None:
                select_negsamples_asposkmer(train_clean_pos_file, train_neg_file, train_seled_neg_file)
            else:
                neg_num = count_line_num(train_clean_neg_file)
                if pos_num <= neg_num:
                    select_negsamples_asposkmer(train_clean_pos_file, train_clean_neg_file, train_seled_neg_file)
                    os.remove(train_clean_neg_file)
                else:
                    train_seled_neg_file = train_clean_neg_file

            fname, fext = os.path.splitext(args.train_file)
            if is_filter_fn:
                train_file = fname + "." + modeltype_str + ".denoise_fpnp" + str(iter_c + 1) + fext
            else:
                train_file = fname + "." + modeltype_str + ".denoise_fp" + str(iter_c + 1) + fext
            concat_two_files(train_clean_pos_file, train_seled_neg_file, concated_fp=train_file)
            os.remove(train_seled_neg_file)
        else:
            if train_clean_neg_file is not None:
                os.remove(train_clean_neg_file)
            print("WARING: The denoise module denoised all samples in the train_file!!!")
        os.remove(train_clean_pos_file)
        print("#####concat denoied file, finished!#####")

        if left_ratio >= args.kept_ratio or pos_num == 0:
            break

    os.remove(train_neg_file)
    total_end = time.time()
    print("###### denoised file for training: {}".format(train_file))
    print("###### training totally costs {:.2f} seconds".format(total_end - total_start))


def display_args(args):
    arg_vars = vars(args)
    print("# ===============================================")
    print("## parameters: ")
    for arg_key in arg_vars.keys():
        if arg_key != 'func':
            print("{}:\n\t{}".format(arg_key, arg_vars[arg_key]))
    print("# ===============================================")


def main():
    parser = argparse.ArgumentParser("train cross rank, filter false positive samples (and "
                                     "false negative samples).")
    parser.add_argument('--train_file', type=str, required=True, help="file containing (combined positive and "
                                                                      "negative) samples for training. better been "
                                                                      "balanced in kmer level.")

    parser.add_argument('--is_filter_fn', type=str, default="no", required=False,
                        help="is filter false negative samples, 'yes' or 'no', default no")

    # model input
    parser.add_argument('--model_type', type=str, default="signal_bilstm",
                        choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
                        required=False,
                        help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
                             "'both_bilstm' means to use both seq and signal bilstm, default: signal_bilstm")
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
    parser.add_argument('--pos_weight', type=float, default=1.0, required=False)
    parser.add_argument('--batch_size', type=int, default=512, required=False)
    parser.add_argument('--lr', type=float, default=0.001, required=False)
    parser.add_argument('--epoch_num', type=int, default=3, required=False)
    parser.add_argument('--step_interval', type=int, default=100, required=False)

    parser.add_argument('--iterations', type=int, default=10, required=False)
    parser.add_argument('--rounds', type=int, default=3, required=False)
    parser.add_argument("--score_cf", type=float, default=0.5,
                        required=False,
                        help="score cutoff to keep high quality (which prob>=score_cf) positive samples. "
                             "(0, 0.5], default 0.5")
    parser.add_argument("--kept_ratio", type=float, default=0.99,
                        required=False,
                        help="kept ratio of samples, to end denoise process. default 0.99")
    parser.add_argument("--fst_iter_prob", action="store_true", default=False,
                        help="if output probs of samples after 1st iteration")

    args = parser.parse_args()

    print("[main] start..")
    total_start = time.time()

    display_args(args)
    denoise(args)

    endtime = time.time()
    print("[main] costs {} seconds".format(endtime-total_start))


if __name__ == '__main__':
    main()
