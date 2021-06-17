#!/usr/bin/env bash
# demo cmds for generating training samples
# 1. deepsignal_plant extract (extract features from fast5s)
deepsignal_plant extract --fast5_dir fast5s/ [--corrected_group --basecall_subgroup --reference_path] --methy_label 1 --motifs CHG --mod_loc 0 --write_path samples_CHG.hc_poses_positive.tsv [--nproc] --positions /path/to/file/contatining/high_confidence/positive/sites.tsv
deepsignal_plant extract --fast5_dir fast5s/ [--corrected_group --basecall_subgroup --reference_path] --methy_label 0 --motifs CHG --mod_loc 0 --write_path samples_CHG.hc_poses_negative.tsv [--nproc] --positions /path/to/file/contatining/high_confidence/negative/sites.tsv

# 2. randomly select equally number (e.g., 10m) of positive and negative samples
# the selected positive and negative samples then can be combined and used for training, see step 4.
python /path/to/scripts/randsel_file_rows.py --ori_filepath samples_CHG.hc_poses_positive.tsv --write_filepath samples_CHG.hc_poses_positive.r10m.tsv --num_lines 10000000 --header false &
python /path/to/scripts/randsel_file_rows.py --ori_filepath samples_CHG.hc_poses_negative.tsv --write_filepath samples_CHG.hc_poses_negative.r10m.tsv --num_lines 10000000 --header false &

# 3. extract balanced negative (or positive) samples if needed
# for example, extract balanced negative samples of each kmer as the number of positive samples of the kmer
python /path/to/scripts/balance_samples_of_kmers.py --feafile samples_CHG.hc_poses_negative.tsv --kmer_feafile samples_CHG.hc_poses_positive.r10m.tsv --wfile samples_CHG.hc_poses_negative.b10m.tsv

# 4. combine positive and negative samples for training
# after combining, the combined file can be splited into two files as training/validating set, see step 6.
python /path/to/scripts/concat_two_files.py --fp1 samples_CHG.hc_poses_positive.r10m.tsv --fp2 samples_CHG.hc_poses_negative.b10m.tsv --concated_fp samples_CHG.hc_poses.rb20m.tsv

# 5. denoise positive (and negative) training samples if needed
# this step will generate a file named "samples_CHG.hc_poses.rb20m.*_bilstm.denoise_*.tsv"
CUDA_VISIBLE_DEVICES=0 deepsignal_plant denoise --train_file samples_CHG.hc_poses.rb20m.tsv --is_filter_fn no/yes [--model_type --epoch_num --rounds --iterations]

# 6. split samples for training/validating
# suppose file "samples_CHG.hc_poses.rb20m.*_bilstm.denoise_*.tsv" has 16000000 lines (samples), and we use 160k samples for validation
head -15840000 samples_CHG.hc_poses.rb20m.*_bilstm.denoise_*.tsv > samples_CHG.hc_poses.rb20m.*_bilstm.denoise_*.train.tsv
tail -160000 samples_CHG.hc_poses.rb20m.*_bilstm.denoise_*.tsv > samples_CHG.hc_poses.rb20m.*_bilstm.denoise_*.valid.tsv

# 7. train
CUDA_VISIBLE_DEVICES=0 deepsignal_plant train --train_file samples_CHG.hc_poses.rb20m.*_bilstm.denoise_*.train.tsv --valid_file samples_CHG.hc_poses.rb20m.*_bilstm.denoise_*.valid.tsv --model_dir model.dplant.CHG --step_interval 1000
