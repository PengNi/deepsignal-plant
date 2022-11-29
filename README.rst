deepsignal-plant
================


Release
-------
0.1.7
-----
optimize/test more on multi-gpu support in call_mods module

add scipy==1.7.3 in requirements (floating point error in scipy>=1.9->tombo). fix this in deepsignal-plant, although tombo only is the pre-process tool for this tool.


0.1.6
-----
bug fixes

enable .gz output in extract/call_mods/call_freq modules

multi-gpu support in call_mods module

update requirements/dependences


0.1.5
-----
make sure results of each read be written together in call_mods' output

make `--reference_path` not a required input in *extract* and *call_mods* module


0.1.4
-----
modify *call_freq* module for large genomes,

fix bug of extrating contig name from fast5s



0.1.3
-----
add ranger optimizer and modify train module,

fix Queue.qsize() NotImplementedError in macOS partially, however *call_mods* in CUDA mode in macOS still doesn't work,

add `init_model` option in train module



0.1.2
-----
change imports in ref_reader,

change requirements,

fix and modify denoise module,

fix MKL_THREADING_LAYER error temporarily,

add `--region` option,

a combined 5mC model replacing CG/CHG/CHH models


0.1.1
-----
fix bug and optimize call_mods without GPU, add call_freq funciton


0.1.0
-----
Release the first vesrion of deepsignal-plant package