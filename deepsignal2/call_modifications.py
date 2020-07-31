"""
call modifications from fast5 files or extracted features,
using tensorflow and the trained model.
output format: chromosome, pos, strand, pos_in_strand, read_name, read_strand,
prob_0, prob_1, called_label, seq
"""

from __future__ import absolute_import

import torch
from torch.utils.data import TensorDataset
import argparse
import os
import sys
import numpy as np
from sklearn import metrics

# import multiprocessing as mp
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

# from utils.process_utils import Queue
from torch.multiprocessing import Queue
import time

from models import ModelBiLSTM
from utils.process_utils import base2code_dna
from utils.process_utils import code2base_dna
from utils.process_utils import str2bool
from utils.process_utils import display_args

from extract_features import _extract_features
from extract_features import _extract_preprocess

from utils.constants_torch import use_cuda

queen_size_border = 1000
time_wait = 5


def _read_features_file(features_file, features_batch_q, batch_num=512):
    with open(features_file, "r") as rf:
        sampleinfo = []  # contains: chromosome, pos, strand, pos_in_strand, read_name, read_strand
        kmers = []
        base_means = []
        base_stds = []
        base_signal_lens = []
        cent_signals = []
        labels = []

        for line in rf:
            words = line.strip().split("\t")

            sampleinfo.append("\t".join(words[0:6]))

            kmers.append([base2code_dna[x] for x in words[6]])
            base_means.append([float(x) for x in words[7].split(",")])
            base_stds.append([float(x) for x in words[8].split(",")])
            base_signal_lens.append([int(x) for x in words[9].split(",")])
            cent_signals.append([float(x) for x in words[10].split(",")])
            labels.append(int(words[11]))

            if len(sampleinfo) == batch_num:
                features_batch_q.put((sampleinfo, kmers, base_means, base_stds,
                                      base_signal_lens, cent_signals, labels))
                while features_batch_q.qsize() > queen_size_border:
                    time.sleep(time_wait)
                sampleinfo = []
                kmers = []
                base_means = []
                base_stds = []
                base_signal_lens = []
                cent_signals = []
                labels = []
        if len(sampleinfo) > 0:
            features_batch_q.put((sampleinfo, kmers, base_means, base_stds,
                                  base_signal_lens, cent_signals, labels))
    features_batch_q.put("kill")


def _call_mods(features_batch, model):
    sampleinfo, kmers, base_means, base_stds, base_signal_lens, \
        cent_signals, labels = features_batch
    labels = np.reshape(labels, (len(labels)))

    testdata = TensorDataset(torch.FloatTensor(kmers), torch.FloatTensor(base_means),
                             torch.FloatTensor(base_stds), torch.FloatTensor(base_signal_lens),
                             torch.FloatTensor(cent_signals))
    testloader = torch.utils.data.DataLoader(testdata, batch_size=labels.shape[0], shuffle=False)

    predicted, logits = None, None
    for vi, vsfeatures in enumerate(testloader):
        # only 1 loop
        vkmer, vbase_means, vbase_stds, vbase_signal_lens, vcent_signals = vsfeatures
        if use_cuda:
            vkmer = vkmer.cuda()
            vbase_means = vbase_means.cuda()
            vbase_stds = vbase_stds.cuda()
            vbase_signal_lens = vbase_signal_lens.cuda()
            vcent_signals = vcent_signals.cuda()
        voutputs, vlogits = model(vkmer, vbase_means, vbase_stds, vbase_signal_lens, vcent_signals)
        _, vpredicted = torch.max(vlogits.data, 1)
        if use_cuda:
            vlogits = vlogits.cpu()
            vpredicted = vpredicted.cpu()

        predicted = vpredicted.numpy()
        logits = vlogits.data.numpy()

    accuracy = metrics.accuracy_score(
        y_true=labels, y_pred=predicted)

    pred_str = []
    for idx in range(labels.shape[0]):
        # chromosome, pos, strand, pos_in_strand, read_name, read_strand, prob_0, prob_1, called_label, seq
        prob_0, prob_1 = logits[idx][0], logits[idx][1]
        prob_0_norm = prob_0 / (prob_0 + prob_1)
        prob_1_norm = prob_1 / (prob_0 + prob_1)
        pred_str.append("\t".join([sampleinfo[idx], str(prob_0_norm),
                                   str(prob_1_norm), str(predicted[idx]),
                                   ''.join([code2base_dna[x] for x in kmers[idx]])]))

    return pred_str, accuracy


def _call_mods_q(model_path, features_batch_q, pred_str_q, success_file, args):
    model = ModelBiLSTM(args.seq_len, args.signal_len, args.layernum_seq, args.layernum_signal,
                        args.class_num, args.dropout_rate, args.hid_rnn,
                        args.n_vocab, args.n_embed, str2bool(args.is_base), str2bool(args.is_signallen),
                        args.out_channels,
                        args.model_type)
    if use_cuda:
        model = model.cuda()

    para_dict = torch.load(model_path)
    model_dict = model.state_dict()
    model_dict.update(para_dict)
    model.load_state_dict(model_dict)

    model.eval()

    accuracy_list = []
    count = 0
    while True:
        if os.path.exists(success_file):
            break

        if features_batch_q.empty():
            time.sleep(time_wait)
            continue

        features_batch = features_batch_q.get()
        if features_batch == "kill":
            open(success_file, 'w').close()
            break

        pred_str, accuracy = _call_mods(features_batch, model)

        pred_str_q.put(pred_str)
        accuracy_list.append(accuracy)
        count += 1
    # print('total accuracy in process {}: {}'.format(os.getpid(), np.mean(accuracy_list)))
    print('call_mods process {} ending, proceed {} batches'.format(os.getpid(), count))


def _write_predstr_to_file(write_fp, predstr_q):
    with open(write_fp, 'w') as wf:
        while True:
            # during test, it's ok without the sleep()
            if predstr_q.empty():
                time.sleep(time_wait)
                continue
            pred_str = predstr_q.get()
            if pred_str == "kill":
                break
            for one_pred_str in pred_str:
                wf.write(one_pred_str + "\n")
            wf.flush()


def _read_features_from_fast5s(fast5s, motif_seqs, chrom2len, positions, args):
    features_list, error = _extract_features(fast5s, args.corrected_group, args.basecall_subgroup,
                                             args.normalize_method, motif_seqs, args.mod_loc, chrom2len,
                                             args.seq_len, args.signal_len,
                                             args.methy_label, positions)
    features_batches = []
    for i in np.arange(0, len(features_list), args.batch_size):
        sampleinfo = []  # contains: chromosome, pos, strand, pos_in_strand, read_name, read_strand
        kmers = []
        base_means = []
        base_stds = []
        base_signal_lens = []
        cent_signals = []
        labels = []

        for features in features_list[i:(i + args.batch_size)]:
            chrom, pos, alignstrand, loc_in_ref, readname, strand, k_mer, signal_means, signal_stds, \
                signal_lens, kmer_cent_signals, f_methy_label = features

            sampleinfo.append("\t".join([chrom, str(pos), alignstrand, str(loc_in_ref), readname, strand]))
            kmers.append([base2code_dna[x] for x in k_mer])
            base_means.append(signal_means)
            base_stds.append(signal_stds)
            base_signal_lens.append(signal_lens)
            cent_signals.append(kmer_cent_signals)
            labels.append(f_methy_label)
        if len(sampleinfo) > 0:
            features_batches.append((sampleinfo, kmers, base_means, base_stds,
                                     base_signal_lens, cent_signals, labels))
    return features_batches, error


def _read_features_fast5s_q(fast5s_q, features_batch_q, errornum_q,
                            motif_seqs, chrom2len, positions, args):
    while not fast5s_q.empty():
        try:
            fast5s = fast5s_q.get()
        except Exception:
            break
        features_batches, error = _read_features_from_fast5s(fast5s, motif_seqs, chrom2len, positions,
                                                             args)
        errornum_q.put(error)
        for features_batch in features_batches:
            features_batch_q.put(features_batch)
        while features_batch_q.qsize() > queen_size_border:
            time.sleep(time_wait)


def _call_mods_from_fast5s_gpu(motif_seqs, chrom2len, fast5s_q, len_fast5s, positions,
                               model_path, success_file,
                               args):
    # features_batch_q = mp.Queue()
    # errornum_q = mp.Queue()
    features_batch_q = Queue()
    errornum_q = Queue()

    # pred_str_q = mp.Queue()
    pred_str_q = Queue()

    nproc = args.nproc
    if nproc < 2:
        nproc = 2
    elif nproc > 2:
        nproc -= 1

    features_batch_procs = []
    for _ in range(nproc - 1):
        p = mp.Process(target=_read_features_fast5s_q, args=(fast5s_q, features_batch_q, errornum_q,
                                                             motif_seqs, chrom2len, positions,
                                                             args))
        p.daemon = True
        p.start()
        features_batch_procs.append(p)

    p_call_mods_gpu = mp.Process(target=_call_mods_q, args=(model_path, features_batch_q, pred_str_q,
                                                            success_file, args))
    p_call_mods_gpu.daemon = True
    p_call_mods_gpu.start()

    print("write_process started..")
    p_w = mp.Process(target=_write_predstr_to_file, args=(args.result_file, pred_str_q))
    p_w.daemon = True
    p_w.start()

    errornum_sum = 0
    while True:
        running = any(p.is_alive() for p in features_batch_procs)
        while not errornum_q.empty():
            errornum_sum += errornum_q.get()
        if not running:
            break

    for p in features_batch_procs:
        p.join()
    features_batch_q.put("kill")

    p_call_mods_gpu.join()

    print("finishing the write_process..")
    pred_str_q.put("kill")

    p_w.join()

    print("%d of %d fast5 files failed.." % (errornum_sum, len_fast5s))


def _fast5s_q_to_pred_str_q(fast5s_q, errornum_q, pred_str_q,
                            motif_seqs, chrom2len, model_path, positions, args):
    model = ModelBiLSTM(args.seq_len, args.signal_len, args.layernum_seq, args.layernum_signal,
                        args.class_num, args.dropout_rate, args.hid_rnn,
                        args.n_vocab, args.n_embed, str2bool(args.is_base), str2bool(args.is_signallen),
                        args.out_channels,
                        args.model_type)
    # this function is designed for CPU, disable cuda
    # if use_cuda:
    #     model = model.cuda()

    para_dict = torch.load(model_path)
    model_dict = model.state_dict()
    model_dict.update(para_dict)
    model.load_state_dict(model_dict)

    model.eval()

    accuracy_list = []
    count = 0

    while not fast5s_q.empty():
        try:
            fast5s = fast5s_q.get()
        except Exception:
            break
        features_batches, error = _read_features_from_fast5s(fast5s, motif_seqs, chrom2len, positions,
                                                             args)
        errornum_q.put(error)
        for features_batch in features_batches:
            pred_str, accuracy = _call_mods(features_batch, model)

            pred_str_q.put(pred_str)
            accuracy_list.append(accuracy)
            count += 1
    # print('total accuracy in process {}: {}'.format(os.getpid(), np.mean(accuracy_list)))
    print('call_mods process {} ending, proceed {} batches'.format(os.getpid(), count))


def _call_mods_from_fast5s_cpu(motif_seqs, chrom2len, fast5s_q, len_fast5s, positions, model_path,
                               success_file, args):

    # errornum_q = mp.Queue()
    errornum_q = Queue()

    # pred_str_q = mp.Queue()
    pred_str_q = Queue()

    nproc = args.nproc
    if nproc > 1:
        nproc -= 1

    pred_str_procs = []
    for _ in range(nproc):
        p = mp.Process(target=_fast5s_q_to_pred_str_q, args=(fast5s_q, errornum_q, pred_str_q,
                                                             motif_seqs, chrom2len, model_path, positions,
                                                             args))
        p.daemon = True
        p.start()
        pred_str_procs.append(p)

    print("write_process started..")
    p_w = mp.Process(target=_write_predstr_to_file, args=(args.result_file, pred_str_q))
    p_w.daemon = True
    p_w.start()

    errornum_sum = 0
    while True:
        running = any(p.is_alive() for p in pred_str_procs)
        while not errornum_q.empty():
            errornum_sum += errornum_q.get()
        if not running:
            break

    for p in pred_str_procs:
        p.join()

    print("finishing the write_process..")
    pred_str_q.put("kill")

    p_w.join()

    print("%d of %d fast5 files failed.." % (errornum_sum, len_fast5s))


def call_mods(args):
    start = time.time()

    model_path = os.path.abspath(args.model_path)
    input_path = os.path.abspath(args.input_path)
    success_file = input_path.rstrip("/") + ".success"
    if os.path.exists(success_file):
        os.remove(success_file)

    if os.path.isdir(input_path):
        motif_seqs, chrom2len, fast5s_q, len_fast5s, positions = _extract_preprocess(input_path,
                                                                                     str2bool(args.recursively),
                                                                                     args.motifs,
                                                                                     str2bool(args.is_dna),
                                                                                     args.reference_path,
                                                                                     args.f5_batch_num,
                                                                                     args.positions)
        if use_cuda:
            _call_mods_from_fast5s_gpu(motif_seqs, chrom2len, fast5s_q, len_fast5s, positions, model_path,
                                       success_file, args)
        else:
            _call_mods_from_fast5s_cpu(motif_seqs, chrom2len, fast5s_q, len_fast5s, positions, model_path,
                                       success_file, args)
    else:
        # features_batch_q = mp.Queue()
        features_batch_q = Queue()
        p_rf = mp.Process(target=_read_features_file, args=(input_path, features_batch_q,
                                                            args.batch_size))
        p_rf.daemon = True
        p_rf.start()

        # pred_str_q = mp.Queue()
        pred_str_q = Queue()

        predstr_procs = []
        nproc = args.nproc
        if nproc > 2:
            nproc -= 2

        if use_cuda:
            nproc_dp = 1
        else:
            nproc_dp = nproc

        for _ in range(nproc_dp):
            p = mp.Process(target=_call_mods_q, args=(model_path, features_batch_q, pred_str_q,
                                                      success_file, args))
            p.daemon = True
            p.start()
            predstr_procs.append(p)

        print("write_process started..")
        p_w = mp.Process(target=_write_predstr_to_file, args=(args.result_file, pred_str_q))
        p_w.daemon = True
        p_w.start()

        for p in predstr_procs:
            p.join()

        print("finishing the write_process..")
        pred_str_q.put("kill")

        p_rf.join()

        p_w.join()

    if os.path.exists(success_file):
        os.remove(success_file)
    print("call_mods costs %.2f seconds.." % (time.time() - start))


def main():
    parser = argparse.ArgumentParser("call modifications")

    p_input = parser.add_argument_group("INPUT")
    p_input.add_argument("--input_path", "-i", action="store", type=str,
                         required=True,
                         help="the input path, can be a signal_feature file from extract_features.py, "
                              "or a directory of fast5 files. If a directory of fast5 files is provided, "
                              "args in FAST5_EXTRACTION should (reference_path must) be provided.")

    p_call = parser.add_argument_group("CALL")
    p_call.add_argument("--model_path", "-m", action="store", type=str, required=True,
                        help="file path of the trained model (.ckpt)")

    # model input
    p_call.add_argument('--model_type', type=str, default="both",
                        choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
                        required=False,
                        help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
                             "'both_bilstm' means to use both seq and signal bilstm, default: both_bilstm")
    p_call.add_argument('--seq_len', type=int, default=11, required=False)
    p_call.add_argument('--signal_len', type=int, default=128, required=False)

    # model param
    p_call.add_argument('--layernum_seq', type=int, default=3,
                        required=False, help="layer num for Sequence Feature, default 3")
    p_call.add_argument('--layernum_signal', type=int, default=2,
                        required=False, help="layer num for Signal Feature, default 2")
    p_call.add_argument('--class_num', type=int, default=2, required=False)
    p_call.add_argument('--dropout_rate', type=float, default=0, required=False)
    p_call.add_argument('--n_vocab', type=int, default=16, required=False,
                        help="base_seq vocab_size (15 base kinds from iupac)")
    p_call.add_argument('--n_embed', type=int, default=4, required=False,
                        help="base_seq embedding_size")
    p_call.add_argument('--is_base', type=str, default="yes", required=False,
                        help="is using base features, default yes")
    p_call.add_argument('--is_signallen', type=str, default="yes", required=False,
                        help="is using signal length feature of each base, default yes")

    p_call.add_argument("--batch_size", "-b", default=512, type=int, required=False,
                        action="store", help="batch size, default 512")

    # BiLSTM model param
    p_call.add_argument('--hid_rnn', type=int, default=256, required=False,
                        help="BiLSTM hidden_size")
    parser.add_argument('--out_channels', type=int, default=128, required=False,
                        help="SignalBiLSTM out_channels")

    # transformer model param
    p_call.add_argument('--d_model', type=int, default=256, required=False)
    p_call.add_argument('--hid_trans', type=int, default=512, required=False,
                        help="transfomer encoder hidden size")
    p_call.add_argument('--n_head', type=int, default=4, required=False)

    p_output = parser.add_argument_group("OUTPUT")
    p_output.add_argument("--result_file", "-o", action="store", type=str, required=True,
                          help="the file path to save the predicted result")

    p_f5 = parser.add_argument_group("FAST5_EXTRACTION")
    p_f5.add_argument("--recursively", "-r", action="store", type=str, required=False,
                      default='yes', help='is to find fast5 files from fast5 dir recursively. '
                                          'default true, t, yes, 1')
    p_f5.add_argument("--corrected_group", action="store", type=str, required=False,
                      default='RawGenomeCorrected_000',
                      help='the corrected_group of fast5 files after '
                           'tombo re-squiggle. default RawGenomeCorrected_000')
    p_f5.add_argument("--basecall_subgroup", action="store", type=str, required=False,
                      default='BaseCalled_template',
                      help='the corrected subgroup of fast5 files. default BaseCalled_template')
    p_f5.add_argument("--reference_path", action="store",
                      type=str, required=False,
                      help="the reference file to be used, usually is a .fa file")
    p_f5.add_argument("--is_dna", action="store", type=str, required=False,
                      default='yes',
                      help='whether the fast5 files from DNA sample or not. '
                           'default true, t, yes, 1. '
                           'setting this option to no/false/0 means '
                           'the fast5 files are from RNA sample.')
    p_f5.add_argument("--normalize_method", action="store", type=str, choices=["mad", "zscore"],
                      default="mad", required=False,
                      help="the way for normalizing signals in read level. "
                           "mad or zscore, default mad")
    p_f5.add_argument("--methy_label", action="store", type=int,
                      choices=[1, 0], required=False, default=1,
                      help="the label of the interested modified bases, this is for training."
                           " 0 or 1, default 1")
    p_f5.add_argument("--motifs", action="store", type=str,
                      required=False, default='CG',
                      help='motif seq to be extracted, default: CG. '
                           'can be multi motifs splited by comma '
                           '(no space allowed in the input str), '
                           'or use IUPAC alphabet, '
                           'the mod_loc of all motifs must be '
                           'the same')
    p_f5.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                      help='0-based location of the targeted base in the motif, default 0')
    p_f5.add_argument("--f5_batch_num", action="store", type=int, default=100,
                      required=False,
                      help="number of files to be processed by each process one time, default 100")
    p_f5.add_argument("--positions", action="store", type=str,
                      required=False, default=None,
                      help="file with a list of positions interested (must be formatted as tab-separated file"
                           " with chromosome, position (in fwd strand), and strand. motifs/mod_loc are still "
                           "need to be set. --positions is used to narrow down the range of the trageted "
                           "motif locs. default None")

    parser.add_argument("--nproc", "-p", action="store", type=int, default=1,
                        required=False, help="number of processes to be used, default 1.")
    # parser.add_argument("--is_gpu", action="store", type=str, default="no", required=False,
    #                     choices=["yes", "no"], help="use gpu for tensorflow or not, default no. "
    #                                                 "If you're using a gpu machine, please set to yes. "
    #                                                 "Note that when is_gpu is yes, --nproc is not valid "
    #                                                 "to tensorflow.")

    args = parser.parse_args()
    display_args(args)

    call_mods(args)


if __name__ == '__main__':
    sys.exit(main())
