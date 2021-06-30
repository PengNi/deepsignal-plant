# DeepSignal-plant

[![Python](https://img.shields.io/pypi/pyversions/deepsignal-plant)](https://www.python.org/)
[![PyPI version](https://img.shields.io/pypi/v/deepsignal-plant)](https://pypi.org/project/deepsignal-plant/)
[![GitHub License](https://img.shields.io/github/license/PengNi/deepsignal-plant)](https://github.com/PengNi/deepsignal-plant/blob/master/LICENSE)
[![PyPI-Downloads](https://pepy.tech/badge/deepsignal-plant)](https://pepy.tech/project/deepsignal-plant)
[![PyPI-Downloads/m](https://pepy.tech/badge/deepsignal-plant/month)](https://pepy.tech/project/deepsignal-plant)

## A deep-learning method for detecting methylation state from Oxford Nanopore sequencing reads of plants.
deepsignal-plant applys BiLSTM to detect methylation from Nanopore reads. It is built on **Python3** and **PyTorch**.

## Contents
- [Installation](#Installation)
- [Trained models](#Trained-models)
- [Example data](#Example-data)
- [Quick start](#Quick-start)
- [Usage](#Usage)

## Installation
deepsignal-plant is built on [Python3](https://www.python.org/) and [PyTorch](https://pytorch.org/). [tombo](https://github.com/nanoporetech/tombo) is required to re-squiggle the raw signals from nanopore reads before running deepsignal-plant.
   - Prerequisites:\
       [Python3.*](https://www.python.org/)\
       [tombo](https://github.com/nanoporetech/tombo) (version 1.5.1)
   - Dependencies:\
       [numpy](http://www.numpy.org/)\
       [h5py](https://github.com/h5py/h5py)\
       [statsmodels](https://github.com/statsmodels/statsmodels/)\
       [scikit-learn](https://scikit-learn.org/stable/)\
       [PyTorch](https://pytorch.org/) (version >=1.2.0, <=1.6.0?)

#### 1. Create an environment
We highly recommend to use a virtual environment for the installation of deepsignal-plant and its dependencies. A virtual environment can be created and (de)activated as follows by using [conda](https://conda.io/docs/):
```bash
# create
conda create -n deepsignalpenv python=3.6
# activate
conda activate deepsignalpenv
# deactivate
conda deactivate
```
The virtual environment can also be created by using [virtualenv](https://github.com/pypa/virtualenv/).

#### 2. Install deepsignal-plant
- After creating and activating the environment, download deepsignal-plant (**lastest version**) from github:
```bash
git clone https://github.com/PengNi/deepsignal-plant.git
cd deepsignal-plant
python setup.py install
```
**or** install deepsignal-plant using *pip*:
```bash
pip install deepsignal-plant
```

- [PyTorch](https://pytorch.org/) can be automatically installed during the installation of deepsignal-plant. However, if the version of [PyTorch](https://pytorch.org/) installed is not appropriate for your OS, an appropriate version should be re-installed in the same environment as the [instructions](https://pytorch.org/get-started/locally/):
```bash
# install using conda
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
# or install using pip
pip install torch==1.4.0 torchvision==0.5.0
```

- [tombo](https://github.com/nanoporetech/tombo) is required to be installed in the same environment:
```bash
# install using conda
conda install -c bioconda ont-tombo
# or install using pip
pip install ont-tombo
```


## Trained models

Currently we have trained the following models:
   * _[model.dp2.CNN.arabnrice2-1_120m_R9.4plus_tem.bn13_sn16.both_bilstm.epoch6.ckpt](https://drive.google.com/file/d/1HnDKPEfCAXgo7vPN-zaD44Kqz1SDw160/view?usp=sharing)_: A 5mC model trained using _A. thaliana_ and _O. sativa_ R9.4 1D reads.
   * (deprecated) ~~_[model.dp2.CG.arabnrice2-1_R9.4plus_tem.bn13_sn16.balance.both_bilstm.b13_s16_epoch6.ckpt](https://drive.google.com/file/d/1BRThGKdRS6cCRXomJ0sGK2vXCFtWbhv7/view?usp=sharing)_: A CpG (5mC) model trained using _A. thaliana_ and _O. sativa_ R9.4 1D reads.~~
   * (deprecated) ~~_[model.dp2.CHG.arabnrice2-1_R9.4plus_tem.bn13_sn16.denoise_signal_bilstm.both_bilstm.b13_s16_epoch4.ckpt](https://drive.google.com/file/d/1mRWIWVvoHnktKzvatvuH59Wg8TFTNSac/view?usp=sharing)_: A CHG (5mC) model trained using _A. thaliana_ and _O. sativa_ R9.4 1D reads.~~
   * (deprecated) ~~_[model.dp2.CHH.arabnrice2-1_R9.4plus_tem.bn13_sn16.denoise_signal_bilstm.both_bilstm.b13_s16_epoch7.ckpt](https://drive.google.com/file/d/1CiIEzhoJvN2OUy4vTfDSe6c2ig6d0j8w/view?usp=sharing)_: A CHH (5mC) model trained using _A. thaliana_ and _O. sativa_ R9.4 1D reads.~~


## Example data

   * _[fast5s.sample.tar.gz](https://drive.google.com/file/d/1PauSQH-3Wpi6FNjNycH9n3GSxkW8C3s0/view?usp=sharing)_: 4000 _A. thaliana_ R9.4 raw reads, with a genome reference.


## Quick start
To call modifications, the raw fast5 files should be basecalled ([Guppy>=3.6.1](https://nanoporetech.com/community)) and then be re-squiggled by [tombo](https://github.com/nanoporetech/tombo). At last, modifications of specified motifs can be called by deepsignal. Belows are commands to call 5mC in CG, CHG, and CHH contexts:
```bash
# Download and unzip the example data and pre-trained models.
# 1. guppy basecall using GPU
guppy_basecaller -i fast5s/ -r -s fast5s_guppy --config dna_r9.4.1_450bps_hac_prom.cfg --device CUDA:0

# 2. tombo resquiggle
cat fast5s_guppy/*.fastq > fast5s_guppy.fastq
tombo preprocess annotate_raw_with_fastqs --fast5-basedir fast5s/ --fastq-filenames fast5s_guppy.fastq --basecall-group Basecall_1D_000 --basecall-subgroup BaseCalled_template --overwrite --processes 10
tombo resquiggle fast5s/ GCF_000001735.4_TAIR10.1_genomic.fna --processes 10 --corrected-group RawGenomeCorrected_000 --basecall-group Basecall_1D_000 --overwrite

# 3. deepsignal-plant call_mods
# 5mCs in all contexts (CG, CHG, and CHH) can be called at one time
# C
CUDA_VISIBLE_DEVICES=0 deepsignal_plant call_mods --input_path fast5s/ --model_path model.dp2.CNN.arabnrice2-1_120m_R9.4plus_tem.bn13_sn16.both_bilstm.epoch6.ckpt --result_file fast5s.C.call_mods.tsv --corrected_group RawGenomeCorrected_000 --reference_path GCF_000001735.4_TAIR10.1_genomic.fna --motifs C --nproc 30 --nproc_gpu 6
deepsignal_plant call_freq --input_path fast5s.C.call_mods.tsv --result_file fast5s.C.call_mods.frequency.tsv
# split freq file into 3 call_freq files as motif (CG/CHG/CHH)
python /path/to/deepsignal_plant/scripts/split_freq_file_by_5mC_motif.py --freqfile fast5s.C.call_mods.frequency.tsv &
```


## Usage
#### 1. Basecall and re-squiggle
Before run deepsignal, the raw reads should be basecalled ([Guppy>=3.6.1](https://nanoporetech.com/community)) and then be processed by the *re-squiggle* module of [tombo](https://github.com/nanoporetech/tombo).

Note:
- If the fast5 files are in multi-read FAST5 format, please use _multi_to_single_fast5_ command from the [ont_fast5_api package](https://github.com/nanoporetech/ont_fast5_api) to convert the fast5 files before using tombo (Ref to [issue #173](https://github.com/nanoporetech/tombo/issues/173) in [tombo](https://github.com/nanoporetech/tombo)).
```bash
multi_to_single_fast5 -i $multi_read_fast5_dir -s $single_read_fast5_dir -t 30 --recursive
```
- If the basecall results are saved as fastq, run the [*tombo proprecess annotate_raw_with_fastqs*](https://nanoporetech.github.io/tombo/resquiggle.html) command before *re-squiggle*.

For the example data:
```bash
# 1. basecall using GPU
guppy_basecaller -i fast5s/ -r -s fast5s_guppy --config dna_r9.4.1_450bps_hac_prom.cfg --device CUDA:0
# or using CPU
guppy_basecaller -i fast5s/ -r -s fast5s_guppy --config dna_r9.4.1_450bps_hac_prom.cfg

# 2. proprecess fast5 if basecall results are saved in fastq format
cat fast5s_guppy/*.fastq > fast5s_guppy.fastq
tombo preprocess annotate_raw_with_fastqs --fast5-basedir fast5s/ --fastq-filenames fast5s_guppy.fastq --basecall-group Basecall_1D_000 --basecall-subgroup BaseCalled_template --overwrite --processes 10

# 3. resquiggle, cmd: tombo resquiggle $fast5_dir $reference_fa
tombo resquiggle fast5s/ GCF_000001735.4_TAIR10.1_genomic.fna --processes 10 --corrected-group RawGenomeCorrected_000 --basecall-group Basecall_1D_000 --overwrite
```

#### 2. extract features
Features of targeted sites can be extracted for training or testing.

For the example data (By default, deepsignal-plant extracts 13-mer-seq and 13*16-signal features of each CpG motif in reads. Note that the value of *--corrected_group* must be the same as that of *--corrected-group* in [tombo](https://github.com/nanoporetech/tombo).):
```bash
# extract features of all Cs
deepsignal_plant extract -i fast5s --reference_path GCF_000001735.4_TAIR10.1_genomic.fna -o fast5s.C.features.tsv --corrected_group RawGenomeCorrected_000 --nproc 30 --motifs C &
```

The extracted_features file is a tab-delimited text file in the following format:
   - **chrom**: the chromosome name
   - **pos**:   0-based position of the targeted base in the chromosome
   - **strand**:    +/-, the aligned strand of the read to the reference
   - **pos_in_strand**: 0-based position of the targeted base in the aligned strand of the chromosome
   - **readname**:  the read name
   - **read_strand**:   t/c, template or complement
   - **k_mer**: the sequence around the targeted base
   - **signal_means**:  signal means of each base in the kmer
   - **signal_stds**:   signal stds of each base in the kmer
   - **signal_lens**:   lens of each base in the kmer
   - **raw_signals**:  signal values for each base of the kmer, splited by ';'
   - **methy_label**:   0/1, the label of the targeted base, for training

#### 3. call modifications

To call modifications, either the extracted-feature file or **the raw fast5 files (recommended)** can be used as input.

For the example data:
```bash
# call 5mCs for instance

# extracted-feature file as input
deepsignal_plant call_mods --input_path fast5s.C.features.tsv --model_path model.dp2.CNN.arabnrice2-1_120m_R9.4plus_tem.bn13_sn16.both_bilstm.epoch6.ckpt --result_file fast5s.C.call_mods.tsv --nproc 30 --nproc_gpu 6

# fast5 files as input, use CPU
CUDA_VISIBLE_DEVICES=-1 deepsignal_plant call_mods --input_path fast5s/ --model_path model.dp2.CNN.arabnrice2-1_120m_R9.4plus_tem.bn13_sn16.both_bilstm.epoch6.ckpt --result_file fast5s.C.call_mods.tsv --corrected_group RawGenomeCorrected_000 --reference_path GCF_000001735.4_TAIR10.1_genomic.fna --motifs C --nproc 30
# fast5 files as input, use GPU
CUDA_VISIBLE_DEVICES=0 deepsignal_plant call_mods --input_path fast5s/ --model_path model.dp2.CNN.arabnrice2-1_120m_R9.4plus_tem.bn13_sn16.both_bilstm.epoch6.ckpt --result_file fast5s.C.call_mods.tsv --corrected_group RawGenomeCorrected_000 --reference_path GCF_000001735.4_TAIR10.1_genomic.fna --motifs C --nproc 30 --nproc_gpu 6
```

The modification_call file is a tab-delimited text file in the following format:
   - **chrom**: the chromosome name
   - **pos**:   0-based position of the targeted base in the chromosome
   - **strand**:    +/-, the aligned strand of the read to the reference
   - **pos_in_strand**: 0-based position of the targeted base in the aligned strand of the chromosome
   - **readname**:  the read name
   - **read_strand**:   t/c, template or complement
   - **prob_0**:    [0, 1], the probability of the targeted base predicted as 0 (unmethylated)
   - **prob_1**:    [0, 1], the probability of the targeted base predicted as 1 (methylated)
   - **called_label**:  0/1, unmethylated/methylated
   - **k_mer**:   the kmer around the targeted base

#### 4. call frequency of modifications
A modification-frequency file can be generated by `call_freq` function with the call_mods file as input:
```bash
# call 5mCs for instance

# output in tsv format
deepsignal_plant call_freq --input_path fast5s.C.call_mods.tsv --result_file fast5s.C.call_mods.frequency.tsv
# output in bedMethyl format
deepsignal_plant call_freq --input_path fast5s.C.call_mods.tsv --result_file fast5s.C.call_mods.frequency.bed --bed
# use --sort to sort the results
deepsignal_plant call_freq --input_path fast5s.C.call_mods.tsv --result_file fast5s.C.call_mods.frequency.bed --bed --sort
```

The modification_frequency file can be either saved in [bedMethyl](https://www.encodeproject.org/data-standards/wgbs/) format (by setting `--bed` as above), or saved as a tab-delimited text file in the following format by default:
   - **chrom**: the chromosome name
   - **pos**:   0-based position of the targeted base in the chromosome
   - **strand**:    +/-, the aligned strand of the read to the reference
   - **pos_in_strand**: 0-based position of the targeted base in the aligned strand of the chromosome
   - **prob_0_sum**:    sum of the probabilities of the targeted base predicted as 0 (unmethylated)
   - **prob_1_sum**:    sum of the probabilities of the targeted base predicted as 1 (methylated)
   - **count_modified**:    number of reads in which the targeted base counted as modified
   - **count_unmodified**:  number of reads in which the targeted base counted as unmodified
   - **coverage**:  number of reads aligned to the targeted base
   - **modification_frequency**:    modification frequency
   - **k_mer**:   the kmer around the targeted base

#### 5. denoise training samples
```bash
# please use deepsignal_plant denoise -h/--help for instructions
deepsignal_plant denoise --train_file /path/to/train/file
```

#### 6. train new models
A new model can be trained as follows:
```bash
# need to split training samples to two independent datasets for training and validating
# please use deepsignal_plant train -h/--help for instructions
deepsignal_plant train --train_file /path/to/train/file --valid_file /path/to/valid/file --model_dir /dir/to/save/the/new/model
```


License
=========
Copyright (C) 2020 [Jianxin Wang](mailto:jxwang@mail.csu.edu.cn), [Feng Luo](mailto:luofeng@clemson.edu), [Peng Ni](mailto:nipeng@csu.edu.cn)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

[Jianxin Wang](mailto:jxwang@mail.csu.edu.cn), [Peng Ni](mailto:nipeng@csu.edu.cn),
School of Computer Science and Engineering, Central South University, Changsha 410083, China

[Feng Luo](mailto:luofeng@clemson.edu), School of Computing, Clemson University, Clemson, SC 29634, USA
