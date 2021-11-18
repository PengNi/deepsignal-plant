#!/usr/bin/python
from __future__ import absolute_import

import sys
import argparse

from .utils.process_utils import str2bool
from .utils.process_utils import display_args

from ._version import DEEPSIGNAL_PLANT_VERSION


def main_extraction(args):
    from .extract_features import extract_features

    display_args(args)

    fast5_dir = args.fast5_dir
    is_recursive = str2bool(args.recursively)

    corrected_group = args.corrected_group
    basecall_subgroup = args.basecall_subgroup
    normalize_method = args.normalize_method

    reference_path = args.reference_path
    is_dna = str2bool(args.is_dna)
    write_path = args.write_path
    w_is_dir = str2bool(args.w_is_dir)
    w_batch_num = args.w_batch_num

    kmer_len = args.seq_len
    signals_len = args.signal_len
    motifs = args.motifs
    mod_loc = args.mod_loc
    methy_label = args.methy_label
    position_file = args.positions
    regionstr = args.region

    nproc = args.nproc
    f5_batch_size = args.f5_batch_size

    extract_features(fast5_dir, is_recursive, reference_path, is_dna,
                     f5_batch_size, write_path, nproc, corrected_group, basecall_subgroup,
                     normalize_method, motifs, mod_loc, kmer_len, signals_len, methy_label,
                     position_file, regionstr, w_is_dir, w_batch_num)


def main_call_mods(args):
    from .call_modifications import call_mods

    display_args(args)
    call_mods(args)


def main_call_freq(args):
    import os
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    from .call_mods_freq import call_mods_frequency_to_file
    display_args(args)
    call_mods_frequency_to_file(args)


def main_train(args):
    from .train import train
    import time

    print("[main] start..")
    total_start = time.time()

    display_args(args)
    train(args)

    endtime = time.time()
    print("[main] costs {} seconds".format(endtime - total_start))


def main_denoise(args):
    from .denoise import denoise
    import time

    print("[main] start..")
    total_start = time.time()

    display_args(args)
    denoise(args)

    endtime = time.time()
    print("[main] costs {} seconds".format(endtime - total_start))


def main():
    parser = argparse.ArgumentParser(prog='deepsignal_plant',
                                     description="detecting base modifications from Nanopore sequencing reads of "
                                                 "plants, "
                                                 "deepsignal_plant contains four modules:\n"
                                                 "\t%(prog)s call_mods: call modifications\n"
                                                 "\t%(prog)s call_freq: call frequency of modifications "
                                                 "at genome level\n"
                                                 "\t%(prog)s extract: extract features from corrected (tombo) "
                                                 "fast5s for training or testing\n"
                                                 "\t%(prog)s train: train a model, need two independent "
                                                 "datasets for training and validating\n"
                                                 "\t%(prog)s denoise: denoise training samples by deep-learning, "
                                                 "filter false positive samples (and false negative samples)",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-v', '--version', action='version',
        version='deepsignal-plant version: {}'.format(DEEPSIGNAL_PLANT_VERSION),
        help='show deepsignal-plant version and exit.')

    subparsers = parser.add_subparsers(title="modules", help='deepsignal_plant modules, use -h/--help for help')
    sub_call_mods = subparsers.add_parser("call_mods", description="call modifications")
    sub_call_freq = subparsers.add_parser("call_freq", description="call frequency of modifications at genome level")
    sub_extract = subparsers.add_parser("extract", description="extract features from corrected (tombo) fast5s for "
                                                               "training or testing."
                                                               "\nIt is suggested that running this module 1 flowcell "
                                                               "a time, or a group of flowcells a time, "
                                                               "if the whole data is extremely large.")
    sub_train = subparsers.add_parser("train", description="train a model, need two independent datasets for training "
                                                           "and validating")
    sub_denoise = subparsers.add_parser("denoise", description="denoise training samples by deep-learning, "
                                                               "filter false positive samples (and "
                                                               "false negative samples).")

    # sub_extract ============================================================================
    se_input = sub_extract.add_argument_group("INPUT")
    se_input.add_argument("--fast5_dir", "-i", action="store", type=str,
                          required=True,
                          help="the directory of fast5 files")
    se_input.add_argument("--recursively", "-r", action="store", type=str, required=False,
                          default='yes',
                          help='is to find fast5 files from fast5_dir recursively. '
                               'default true, t, yes, 1')
    se_input.add_argument("--corrected_group", action="store", type=str, required=False,
                          default='RawGenomeCorrected_000',
                          help='the corrected_group of fast5 files after '
                               'tombo re-squiggle. default RawGenomeCorrected_000')
    se_input.add_argument("--basecall_subgroup", action="store", type=str, required=False,
                          default='BaseCalled_template',
                          help='the corrected subgroup of fast5 files. default BaseCalled_template')
    se_input.add_argument("--reference_path", action="store",
                          type=str, required=True,
                          help="the reference file to be used, usually is a .fa file")
    se_input.add_argument("--is_dna", action="store", type=str, required=False,
                          default='yes',
                          help='whether the fast5 files from DNA sample or not. '
                               'default true, t, yes, 1. '
                               'set this option to no/false/0 if '
                               'the fast5 files are from RNA sample.')

    se_extraction = sub_extract.add_argument_group("EXTRACTION")
    se_extraction.add_argument("--normalize_method", action="store", type=str, choices=["mad", "zscore"],
                               default="mad", required=False,
                               help="the way for normalizing signals in read level. "
                                    "mad or zscore, default mad")
    se_extraction.add_argument("--methy_label", action="store", type=int,
                               choices=[1, 0], required=False, default=1,
                               help="the label of the interested modified bases, this is for training."
                                    " 0 or 1, default 1")
    se_extraction.add_argument("--seq_len", action="store",
                               type=int, required=False, default=13,
                               help="len of kmer. default 13")
    se_extraction.add_argument("--signal_len", action="store",
                               type=int, required=False, default=16,
                               help="the number of signals of one base to be used in deepsignal_plant, default 16")
    se_extraction.add_argument("--motifs", action="store", type=str,
                               required=False, default='CG',
                               help='motif seq to be extracted, default: CG. '
                                    'can be multi motifs splited by comma '
                                    '(no space allowed in the input str), '
                                    'or use IUPAC alphabet, '
                                    'the mod_loc of all motifs must be '
                                    'the same')
    se_extraction.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                               help='0-based location of the targeted base in the motif, default 0')
    se_extraction.add_argument("--region", action="store", type=str,
                               required=False, default=None,
                               help="region of interest, e.g.: chr1, chr1:0, chr1:0-10000. "
                                    "0-based, half-open interval: [start, end). "
                                    "default None, means processing all sites in genome")
    se_extraction.add_argument("--positions", action="store", type=str,
                               required=False, default=None,
                               help="file with a list of positions interested (must be formatted as tab-separated file"
                                    " with chromosome, position (in fwd strand), and strand. motifs/mod_loc are still "
                                    "need to be set. --positions is used to narrow down the range of the trageted "
                                    "motif locs. default None")

    se_output = sub_extract.add_argument_group("OUTPUT")
    se_output.add_argument("--write_path", "-o", action="store",
                           type=str, required=True,
                           help='file path to save the features')
    se_output.add_argument("--w_is_dir", action="store",
                           type=str, required=False, default="no",
                           help='if using a dir to save features into multiple files')
    se_output.add_argument("--w_batch_num", action="store",
                           type=int, required=False, default=200,
                           help='features batch num to save in a single writed file when --is_dir is true')

    sub_extract.add_argument("--nproc", "-p", action="store", type=int, default=1,
                             required=False,
                             help="number of processes to be used, default 1")
    sub_extract.add_argument("--f5_batch_size", action="store", type=int, default=20,
                             required=False,
                             help="number of files to be processed by each process one time, default 20")

    sub_extract.set_defaults(func=main_extraction)

    # sub_call_mods =============================================================================================
    sc_input = sub_call_mods.add_argument_group("INPUT")
    sc_input.add_argument("--input_path", "-i", action="store", type=str,
                          required=True,
                          help="the input path, can be a signal_feature file from extract_features.py, "
                               "or a directory of fast5 files. If a directory of fast5 files is provided, "
                               "args in FAST5_EXTRACTION should (reference_path must) be provided.")

    sc_call = sub_call_mods.add_argument_group("CALL")
    sc_call.add_argument("--model_path", "-m", action="store", type=str, required=True,
                         help="file path of the trained model (.ckpt)")

    # model input
    sc_call.add_argument('--model_type', type=str, default="both_bilstm",
                         choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
                         required=False,
                         help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
                              "'both_bilstm' means to use both seq and signal bilstm, default: both_bilstm")
    sc_call.add_argument('--seq_len', type=int, default=13, required=False,
                         help="len of kmer. default 13")
    sc_call.add_argument('--signal_len', type=int, default=16, required=False,
                         help="signal num of one base, default 16")

    # model param
    sc_call.add_argument('--layernum1', type=int, default=3,
                         required=False, help="lstm layer num for combined feature, default 3")
    sc_call.add_argument('--layernum2', type=int, default=1,
                         required=False, help="lstm layer num for seq feature (and for signal feature too), default 1")
    sc_call.add_argument('--class_num', type=int, default=2, required=False)
    sc_call.add_argument('--dropout_rate', type=float, default=0, required=False)
    sc_call.add_argument('--n_vocab', type=int, default=16, required=False,
                         help="base_seq vocab_size (15 base kinds from iupac)")
    sc_call.add_argument('--n_embed', type=int, default=4, required=False,
                         help="base_seq embedding_size")
    sc_call.add_argument('--is_base', type=str, default="yes", required=False,
                         help="is using base features in seq model, default yes")
    sc_call.add_argument('--is_signallen', type=str, default="yes", required=False,
                         help="is using signal length feature of each base in seq model, default yes")

    sc_call.add_argument("--batch_size", "-b", default=512, type=int, required=False,
                         action="store", help="batch size, default 512")

    # BiLSTM model param
    sc_call.add_argument('--hid_rnn', type=int, default=256, required=False,
                         help="BiLSTM hidden_size for combined feature")

    sc_output = sub_call_mods.add_argument_group("OUTPUT")
    sc_output.add_argument("--result_file", "-o", action="store", type=str, required=True,
                           help="the file path to save the predicted result")

    sc_f5 = sub_call_mods.add_argument_group("FAST5_EXTRACTION")
    sc_f5.add_argument("--recursively", "-r", action="store", type=str, required=False,
                       default='yes', help='is to find fast5 files from fast5 dir recursively. '
                                           'default true, t, yes, 1')
    sc_f5.add_argument("--corrected_group", action="store", type=str, required=False,
                       default='RawGenomeCorrected_000',
                       help='the corrected_group of fast5 files after '
                            'tombo re-squiggle. default RawGenomeCorrected_000')
    sc_f5.add_argument("--basecall_subgroup", action="store", type=str, required=False,
                       default='BaseCalled_template',
                       help='the corrected subgroup of fast5 files. default BaseCalled_template')
    sc_f5.add_argument("--reference_path", action="store",
                       type=str, required=False,
                       help="the reference file to be used, usually is a .fa file")
    sc_f5.add_argument("--is_dna", action="store", type=str, required=False,
                       default='yes',
                       help='whether the fast5 files from DNA sample or not. '
                            'default true, t, yes, 1. '
                            'setting this option to no/false/0 means '
                            'the fast5 files are from RNA sample.')
    sc_f5.add_argument("--normalize_method", action="store", type=str, choices=["mad", "zscore"],
                       default="mad", required=False,
                       help="the way for normalizing signals in read level. "
                            "mad or zscore, default mad")
    # sc_f5.add_argument("--methy_label", action="store", type=int,
    #                    choices=[1, 0], required=False, default=1,
    #                    help="the label of the interested modified bases, this is for training."
    #                         " 0 or 1, default 1")
    sc_f5.add_argument("--motifs", action="store", type=str,
                       required=False, default='CG',
                       help='motif seq to be extracted, default: CG. '
                            'can be multi motifs splited by comma '
                            '(no space allowed in the input str), '
                            'or use IUPAC alphabet, '
                            'the mod_loc of all motifs must be '
                            'the same')
    sc_f5.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                       help='0-based location of the targeted base in the motif, default 0')
    sc_f5.add_argument("--f5_batch_size", action="store", type=int, default=10,
                       required=False,
                       help="number of files to be processed by each process one time, default 10")
    sc_f5.add_argument("--region", action="store", type=str,
                       required=False, default=None,
                       help="region of interest, e.g.: chr1, chr1:0, chr1:0-10000. "
                            "0-based, half-open interval: [start, end). "
                            "default None, means processing the whole sites in genome")
    sc_f5.add_argument("--positions", action="store", type=str,
                       required=False, default=None,
                       help="file with a list of positions interested (must be formatted as tab-separated file"
                            " with chromosome, position (in fwd strand), and strand. motifs/mod_loc are still "
                            "need to be set. --positions is used to narrow down the range of the trageted "
                            "motif locs. default None")

    sub_call_mods.add_argument("--nproc", "-p", action="store", type=int, default=10,
                               required=False, help="number of processes to be used, default 10.")
    sub_call_mods.add_argument("--nproc_gpu", action="store", type=int, default=2,
                               required=False, help="number of processes to use gpu (if gpu is available), "
                                                    "1 or a number less than nproc-1, no more than "
                                                    "nproc/4 is suggested. default 2.")

    sub_call_mods.set_defaults(func=main_call_mods)

    # sub_train =====================================================================================
    st_input = sub_train.add_argument_group("INPUT")
    st_input.add_argument('--train_file', type=str, required=True)
    st_input.add_argument('--valid_file', type=str, required=True)

    st_output = sub_train.add_argument_group("OUTPUT")
    st_output.add_argument('--model_dir', type=str, required=True)

    st_train = sub_train.add_argument_group("TRAIN")
    # model input
    st_train.add_argument('--model_type', type=str, default="both_bilstm",
                          choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
                          required=False,
                          help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
                               "'both_bilstm' means to use both seq and signal bilstm, default: both_bilstm")
    st_train.add_argument('--seq_len', type=int, default=13, required=False,
                          help="len of kmer. default 13")
    st_train.add_argument('--signal_len', type=int, default=16, required=False,
                          help="the number of signals of one base to be used in deepsignal_plant, default 16")
    # model param
    st_train.add_argument('--layernum1', type=int, default=3,
                          required=False, help="lstm layer num for combined feature, default 3")
    st_train.add_argument('--layernum2', type=int, default=1,
                          required=False, help="lstm layer num for seq feature (and for signal feature too), default 1")
    st_train.add_argument('--class_num', type=int, default=2, required=False)
    st_train.add_argument('--dropout_rate', type=float, default=0.5, required=False)
    st_train.add_argument('--n_vocab', type=int, default=16, required=False,
                          help="base_seq vocab_size (15 base kinds from iupac)")
    st_train.add_argument('--n_embed', type=int, default=4, required=False,
                          help="base_seq embedding_size")
    st_train.add_argument('--is_base', type=str, default="yes", required=False,
                          help="is using base features in seq model, default yes")
    st_train.add_argument('--is_signallen', type=str, default="yes", required=False,
                          help="is using signal length feature of each base in seq model, default yes")
    # BiLSTM model param
    st_train.add_argument('--hid_rnn', type=int, default=256, required=False,
                          help="BiLSTM hidden_size for combined feature")
    # model training
    st_train.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "RMSprop", "SGD",
                                                                             "Ranger"],
                          required=False, help="type of optimizer to use, 'Adam' or 'SGD' or 'RMSprop' or 'Ranger', "
                                               "default Adam")
    st_train.add_argument('--batch_size', type=int, default=512, required=False)
    st_train.add_argument('--lr', type=float, default=0.001, required=False)
    st_train.add_argument('--lr_decay', type=float, default=0.1, required=False)
    st_train.add_argument('--lr_decay_step', type=int, default=2, required=False)
    st_train.add_argument("--max_epoch_num", action="store", default=10, type=int,
                          required=False, help="max epoch num, default 10")
    st_train.add_argument("--min_epoch_num", action="store", default=5, type=int,
                          required=False, help="min epoch num, default 5")
    st_train.add_argument('--step_interval', type=int, default=100, required=False)

    st_train.add_argument('--pos_weight', type=float, default=1.0, required=False)
    st_train.add_argument('--init_model', type=str, default=None, required=False,
                          help="file path of pre-trained model parameters to load before training")
    # st_train.add_argument('--seed', type=int, default=1234,
    #                        help='random seed')
    # else
    st_train.add_argument('--tmpdir', type=str, default="/tmp", required=False)

    sub_train.set_defaults(func=main_train)

    # sub_denoise =====================================================================================
    sd_input = sub_denoise.add_argument_group("INPUT")
    sd_input.add_argument('--train_file', type=str, required=True, help="file containing (combined positive and "
                                                                        "negative) samples for training. better been "
                                                                        "balanced in kmer level.")

    sd_train = sub_denoise.add_argument_group("TRAIN")
    sd_train.add_argument('--is_filter_fn', type=str, default="no", required=False,
                          help="is filter false negative samples, , 'yes' or 'no', default no")
    # model input
    sd_train.add_argument('--model_type', type=str, default="signal_bilstm",
                          choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
                          required=False,
                          help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
                               "'both_bilstm' means to use both seq and signal bilstm, default: signal_bilstm")
    sd_train.add_argument('--seq_len', type=int, default=13, required=False,
                          help="len of kmer. default 13")
    sd_train.add_argument('--signal_len', type=int, default=16, required=False,
                          help="the number of signals of one base to be used in deepsignal_plant, default 16")
    # model param
    sd_train.add_argument('--layernum1', type=int, default=3,
                          required=False, help="lstm layer num for combined feature, default 3")
    sd_train.add_argument('--layernum2', type=int, default=1,
                          required=False, help="lstm layer num for seq feature (and for signal feature too), default 1")
    sd_train.add_argument('--class_num', type=int, default=2, required=False)
    sd_train.add_argument('--dropout_rate', type=float, default=0.5, required=False)
    sd_train.add_argument('--n_vocab', type=int, default=16, required=False,
                          help="base_seq vocab_size (15 base kinds from iupac)")
    sd_train.add_argument('--n_embed', type=int, default=4, required=False,
                          help="base_seq embedding_size")
    sd_train.add_argument('--is_base', type=str, default="yes", required=False,
                          help="is using base features in seq model, default yes")
    sd_train.add_argument('--is_signallen', type=str, default="yes", required=False,
                          help="is using signal length feature of each base in seq model, default yes")
    # BiLSTM model param
    sd_train.add_argument('--hid_rnn', type=int, default=256, required=False,
                          help="BiLSTM hidden_size for combined feature")
    # model training
    sd_train.add_argument('--pos_weight', type=float, default=1.0, required=False)
    sd_train.add_argument('--batch_size', type=int, default=512, required=False)
    sd_train.add_argument('--lr', type=float, default=0.001, required=False)
    sd_train.add_argument('--epoch_num', type=int, default=3, required=False)
    sd_train.add_argument('--step_interval', type=int, default=100, required=False)

    sd_denoise = sub_denoise.add_argument_group("DENOISE")
    sd_denoise.add_argument('--iterations', type=int, default=10, required=False)
    sd_denoise.add_argument('--rounds', type=int, default=3, required=False)
    sd_denoise.add_argument("--score_cf", type=float, default=0.5,
                            required=False,
                            help="score cutoff to keep high quality (which prob>=score_cf) positive samples. "
                                 "(0, 0.5], default 0.5")
    sd_denoise.add_argument("--kept_ratio", type=float, default=0.99,
                            required=False,
                            help="kept ratio of samples, to end denoise process. default 0.99")
    sd_denoise.add_argument("--fst_iter_prob", action="store_true", default=False,
                            help="if output probs of samples after 1st iteration")

    sub_denoise.set_defaults(func=main_denoise)

    # sub_call_freq =====================================================================================
    scf_input = sub_call_freq.add_argument_group("INPUT")
    scf_input.add_argument('--input_path', '-i', action="append", type=str, required=True,
                           help='an output file from call_mods/call_modifications.py, or a directory contains '
                                'a bunch of output files. this arg is in "append" mode, can be used multiple times')
    scf_input.add_argument('--file_uid', type=str, action="store", required=False, default=None,
                           help='a unique str which all input files has, this is for finding all input files '
                                'and ignoring the not-input-files in a input directory. if input_path is a file, '
                                'ignore this arg.')

    scf_output = sub_call_freq.add_argument_group("OUTPUT")
    scf_output.add_argument('--result_file', '-o', action="store", type=str, required=True,
                            help='the file path to save the result')

    scf_cal = sub_call_freq.add_argument_group("CAlCULATE")
    scf_cal.add_argument('--bed', action='store_true', default=False, help="save the result in bedMethyl format")
    scf_cal.add_argument('--sort', action='store_true', default=False, help="sort items in the result")
    scf_cal.add_argument('--prob_cf', type=float, action="store", required=False, default=0.5,
                         help='this is to remove ambiguous calls. '
                              'if abs(prob1-prob0)>=prob_cf, then we use the call. e.g., proc_cf=0 '
                              'means use all calls. range [0, 1], default 0.5.')

    scf_para = sub_call_freq.add_argument_group("PARALLEL")
    scf_para.add_argument('--contigs', action="store", type=str, required=False, default=None,
                          help="path of a file contains chromosome/contig names, one name each line; "
                               "or a string contains multiple chromosome names splited by comma. "
                               "default None, which means all chromosomes will be processed at one time. "
                               "If not None, one chromosome will be processed by one subprocess.")
    scf_para.add_argument('--nproc', action="store", type=int, required=False, default=1,
                          help="number of subprocesses used when --contigs is set. i.e., number of contigs processed "
                               "in parallel. default 1")

    sub_call_freq.set_defaults(func=main_call_freq)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    sys.exit(main())
