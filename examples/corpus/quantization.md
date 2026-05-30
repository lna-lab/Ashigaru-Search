# Quantization: NVFP4 and FP8

Quantization shrinks model weights (and sometimes activations) to fewer bits to cut memory and
boost throughput. FP8 (e4m3) keeps 8-bit floats and is a robust default on modern GPUs. NVFP4 is
a 4-bit floating-point format (e2m1) that stores weights in 16-value blocks, each with an FP8
block scale plus an FP32 per-tensor global scale; W4A4 NVFP4 also quantizes activations and needs
a static input scale calibrated on sample data. NVFP4 roughly halves memory versus FP8 and, on
Blackwell, can raise decode throughput, at some accuracy cost that calibration aims to minimize.
