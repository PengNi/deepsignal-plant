deepsignal-plant
================


Release
-------


0.1.4
-----



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