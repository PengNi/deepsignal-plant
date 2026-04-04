import os
import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
import torch.distributed as dist

# pod5 and pyslow5 are lazily imported inside pod5_producer to avoid
# loading their native libraries in every subprocess (resource exhaustion)
from .utils.process_utils import base2code_dna,key_sep
from .utils.process_utils import get_logger
from .utils.process_utils import get_refloc_of_methysite_in_motif
from .utils.process_utils import fill_files_queue
from .extract_features import _get_signals_rect
from .extract_features import _normalize_signals
from .extract_features_pod5 import _group_signals_by_movetable_v2
from .extract_features_pod5 import get_q2tloc_from_cigar
import math
import queue
import time
LOGGER = get_logger(__name__)

def pod5_producer(files_dr, data_queue, num_consumers, format_type):
    """
    负责从文件读取并匹配 BAM，将匹配好的原始数据放入队列
    """
    # 填充文件队列 (假设使用你原来的 fill_files_queue)
    # pod5s_list = []
    # fill_files_queue(pod5s_list, files_dr) 
    
    if format_type == 'pod5':
        import pod5 as _pod5
    elif format_type == 'slow5':
        import pyslow5 as _pyslow5

    for file in files_dr:
        try:
            if format_type == 'pod5':
                with _pod5.Reader(file) as reader:
                    #LOGGER.debug("pod5 file: {}".format(file))
                    for read_record in reader.reads():
                        #LOGGER.debug("read_record")
                        read_name = str(read_record.read_id)
                        
                        data_queue.put((read_name,read_record.signal.copy()))
                        #LOGGER.debug("read_name: {}".format(read_name))
                        #LOGGER.debug("data_queue.put")
                        # try:
                        #     # # 提前匹配 BAM，减少 Dataset 负担
                        #     # for seq_read in bam_index.get_alignments(read_name):
                        #     #     # 只把必要的原始数据塞进队列 (signal + alignment object)
                        #     #     data_queue.put((read_record.signal, seq_read))
                        # except KeyError:
                        #     continue
            elif format_type == 'slow5':
                with _pyslow5.Open(file, 'r') as reader:
                    for read_record in reader.seq_reads():
                        read_name = str(read_record['read_id'])
                        data_queue.put((read_name,read_record['signal'].copy()))
                        # 
                        # try:
                        #     # 提前匹配 BAM，减少 Dataset 负担
                        #     for seq_read in bam_index.get_alignments(read_name):
                        #         # 只把必要的原始数据塞进队列 (signal + alignment object)
                        #         data_queue.put((read_record.signal, seq_read))
                        # except KeyError:
                        #     continue
        except queue.Full:
            #LOGGER.warning("Queue full, waiting...")
            time.sleep(0.1)
        except Exception as e:
            #LOGGER.error(f"Failed to put data: {e}")
            return
    # 重点：放 num_consumers 个 None 信号
    for _ in range(num_consumers):
        data_queue.put(None)
    #stop_event.set()
    print("[Producer] All files parsed and matched.", flush=True)
class SignalDataset(IterableDataset):
    def __init__(self, data_queue,bam_index,motif_seqs,positions, args, device):
        super(SignalDataset).__init__()
        self.data_queue = data_queue
        self.args = args
        self.device = device
        self.bam_index = bam_index
        self.motif_seqs = motif_seqs
        self.positions = positions
        #self.stop_event = stop_event


    def __iter__(self):
        #LOGGER.debug("SignalDataset __iter__")
        for item in iter(self.data_queue.get, None):
            #LOGGER.debug("SignalDataset __iter__ item")
            try:
                # 展开生成器产出的每一个特征包
                for feature in self.process_sig_seq(item):
                    if feature is not None:
                        #LOGGER.debug('feature len: {}'.format(len(feature)))
                        yield feature
            except queue.Empty:
                # 如果队列空了，先检查生产者是否崩溃了
                # if self.stop_event.is_set():
                #     # 生产者已经死了且队列没东西，说明出错了
                #     LOGGER.error("Producer is dead and queue is empty.")                   
                #     return
                time.sleep(0.1)
                #LOGGER.debug("Queue is empty, waiting...")
                continue # 生产者还在跑，只是比较慢，继续等
            except Exception as e:
                #LOGGER.error(f"Error processing read: {e}")
                continue           

    def process_sig_seq(self, item):
        """
        处理单个 Read,改为生成器模式
        """
        #LOGGER.debug("process_sig_seq")
        read_name, signal = item

        if signal is None:
            return

        try:
            # get_alignments 可能会返回多个比对结果（如补充比对）
            for seq_read in self.bam_index.get_alignments(read_name):
                # 内部调用处理逻辑
                seq = seq_read.get_forward_sequence()
                if seq is None:
                    continue
                features_list = self.process_data(signal, seq_read)
                
                # 逐个产出特征，减少内存占用
                for item in features_list:
                    if len(item) > 0:
                        yield to_tensor(item, self.device)
        except KeyError:
            # LOGGER.warning(f"Read:{read_name} not found in BAM")
            pass

    def process_data(self, signal, seq_read):
        """
        原始的特征提取逻辑 (保持逻辑不变，但优化入参)
        """
        #LOGGER.debug("process_data")
        features_list = []
        
        # --- 1. 基础过滤 ---
        # if seq_read.mapping_quality < self.args.mapq:
        #     return features_list
            
        seq = seq_read.get_forward_sequence()
        if seq is None:
            return features_list

        # --- 2. 信号预处理 (移动表解析) ---
        read_dict = dict(seq_read.tags)
        if "mv" not in read_dict:
            return features_list
            
        mv_table = np.asarray(read_dict["mv"][1:])
        stride = int(read_dict["mv"][0])
        num_trimmed = read_dict["ts"]
        if seq_read.has_tag('sp'):
            num_trimmed += seq_read.get_tag('sp')
        
        signal_trimmed = signal[num_trimmed:] if num_trimmed >= 0 else signal[:num_trimmed]
        norm_signals = _normalize_signals(signal_trimmed, self.args.normalize_method)
        signal_group = _group_signals_by_movetable_v2(norm_signals, mv_table, stride)

        # --- 3. 位点筛选与比对坐标映射 ---
        tsite_locs = get_refloc_of_methysite_in_motif(seq, set(self.motif_seqs), self.args.mod_loc)
        
        strand = "."
        q_to_r_poss = None
        if not seq_read.is_unmapped:
            strand = "-" if seq_read.is_reverse else "+"
            strand_code = -1 if seq_read.is_reverse else 1
            ref_start = seq_read.reference_start
            ref_end = seq_read.reference_end
            cigar_tuples = seq_read.cigartuples
            qalign_start = seq_read.query_alignment_start
            qalign_end = seq_read.query_alignment_end
            # if (qalign_end-qalign_start)/seq_read.query_length<self.args.coverage_ratio:
            #     return features_list
            if seq_read.is_reverse:
                seq_start = len(seq) - qalign_end
                seq_end = len(seq) - qalign_start
            else:
                seq_start = qalign_start
                seq_end = qalign_end
            q_to_r_poss = get_q2tloc_from_cigar(
                cigar_tuples, strand_code, (seq_end - seq_start)
            )
        if seq_read.reference_name is None:
            ref_name = "."
        else:
            ref_name = seq_read.reference_name

        
        # --- 4. 提取特征窗口 ---
        num_bases = (self.args.seq_len - 1) // 2
        for loc_idx, loc_in_read in enumerate(tsite_locs):
            # 边界检查
            if num_bases <= loc_in_read < len(seq) - num_bases:
                ref_pos = -1
                if not seq_read.is_unmapped:
                    if seq_start <= loc_in_read < seq_end:
                        offset_idx = loc_in_read - seq_start
                        if q_to_r_poss[offset_idx] != -1:
                            if strand == "-":
                                # pos = '.'#loc_in_read
                                ref_pos = ref_end - 1 - q_to_r_poss[offset_idx]
                            else:
                                # pos = loc_in_read
                                ref_pos = ref_start + q_to_r_poss[offset_idx]
                    else:
                        continue
                if (self.positions is not None) and (
                    key_sep.join([ref_name, str(ref_pos), strand]) not in self.positions
                ):
                    continue
                # 示例：构建特征包
                k_mer = seq[(loc_in_read - num_bases) : (loc_in_read + num_bases + 1)]
                k_seq = [base2code_dna[x] for x in k_mer]
                k_signals = signal_group[(loc_in_read - num_bases) : (loc_in_read + num_bases + 1)]
                
                # 这里的 signal_rect 等计算...
                sampleinfo = "\t".join([
                    seq_read.reference_name or ".", 
                    str(ref_pos), 
                    ("-" if seq_read.is_reverse else "+"), 
                    ".", 
                    seq_read.query_name, 
                    "."
                ])

                features_list.append((
                    sampleinfo,
                    k_seq,
                    [np.mean(x) for x in k_signals],
                    [np.std(x) for x in k_signals],
                    [len(x) for x in k_signals],
                    _get_signals_rect(k_signals, self.args.signal_len),
                    self.args.methy_label,
                ))
                
        return features_list
    
def to_tensor(item, device='cpu'):
    sampleinfo, k_seq, signal_means, signal_stds, signal_lens, k_signals, label = item
    # 转换为张量并进行 reshape
    k_seq = torch.tensor(k_seq, dtype=torch.long)#, device=device)  # (seq_len,)
    signal_means = torch.tensor(signal_means, dtype=torch.float32).view(-1, 1)#, device=device).view(-1, 1)  # (seq_len, 1)
    signal_stds = torch.tensor(signal_stds, dtype=torch.float32).view(-1, 1)#, device=device).view(-1, 1)  # (seq_len, 1)
    signal_lens = torch.tensor(signal_lens, dtype=torch.float32).view(-1, 1)# device=device).view(-1, 1)  # (seq_len, 1)
    k_signals = torch.tensor(k_signals, dtype=torch.float32)# device=device)  # (seq_len, signal_len)
    label = torch.tensor(label, dtype=torch.long)# device=device)  # 标量
    return sampleinfo, k_seq, signal_means, signal_stds, signal_lens, k_signals, label

def worker_init_fn(worker_id: int) -> None:
    pass

def collate_fn_inference(batch):
    
    # ref_name, ref_pos, strand, placeholder1, read_name, placeholder2, k_mer, signal_means, signal_stds, signal_lens, k_signals_rect, methy_label = zip(
    #     *batch)
    sampleinfos, kmers, base_means, base_stds, base_signal_lens, k_signals, labels = zip(*batch)
    # 堆叠成批次张量
    kmers = torch.stack(kmers)  # (batch_size, seq_len)
    base_means = torch.stack(base_means)  # (batch_size, seq_len, 1)
    base_stds = torch.stack(base_stds)  # (batch_size, seq_len, 1)
    base_signal_lens = torch.stack(base_signal_lens)  # (batch_size, seq_len, 1)
    k_signals = torch.stack(k_signals)  # (batch_size, seq_len, signal_len)
    labels = torch.stack(labels)  # (batch_size,)
    #LOGGER.debug("collate_fn_inference")
    return sampleinfos, kmers, base_means, base_stds, base_signal_lens, k_signals, labels
