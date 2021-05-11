import torch
from deepsignal_plant.models import ModelBiLSTM
import argparse
import os


def str2bool(v):
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    p_call = argparse.ArgumentParser("unzip model_ckpt saved by torch 1.6+, for lower version torch use")
    p_call.add_argument("--model_file", type=str, required=True, help="model path")

    p_call.add_argument('--model_type', type=str, default="both_bilstm",
                        choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
                        required=False,
                        help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
                             "'both_bilstm' means to use both seq and signal bilstm, default: both_bilstm")
    p_call.add_argument('--seq_len', type=int, default=13, required=False,
                        help="len of kmer. default 13")
    p_call.add_argument('--signal_len', type=int, default=15, required=False,
                        help="signal num of one base, default 15")

    # model param
    p_call.add_argument('--layernum1', type=int, default=3,
                        required=False, help="lstm layer num for combined feature, default 3")
    p_call.add_argument('--layernum2', type=int, default=1,
                        required=False, help="lstm layer num for seq feature (and for signal feature too), default 1")
    p_call.add_argument('--class_num', type=int, default=2, required=False)
    p_call.add_argument('--dropout_rate', type=float, default=0, required=False)
    p_call.add_argument('--n_vocab', type=int, default=16, required=False,
                        help="base_seq vocab_size (15 base kinds from iupac)")
    p_call.add_argument('--n_embed', type=int, default=4, required=False,
                        help="base_seq embedding_size")
    p_call.add_argument('--is_base', type=str, default="yes", required=False,
                        help="is using base features in seq model, default yes")
    p_call.add_argument('--is_signallen', type=str, default="yes", required=False,
                        help="is using signal length feature of each base in seq model, default yes")
    # BiLSTM model param
    p_call.add_argument('--hid_rnn', type=int, default=256, required=False,
                        help="BiLSTM hidden_size for combined feature")

    args = p_call.parse_args()

    model = ModelBiLSTM(args.seq_len, args.signal_len, args.layernum1, args.layernum2, args.class_num,
                        args.dropout_rate, args.hid_rnn,
                        args.n_vocab, args.n_embed, str2bool(args.is_base), str2bool(args.is_signallen),
                        args.model_type)

    model.load_state_dict(torch.load(args.model_file, map_location=torch.device('cpu')))
    fname, fext = os.path.splitext(args.model_file)
    mode_unzip_file = fname + ".unzip" + fext
    torch.save(model.state_dict(), mode_unzip_file, _use_new_zipfile_serialization=False)
