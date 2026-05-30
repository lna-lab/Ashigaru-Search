# vLLM

vLLM is a high-throughput inference server for large language models. Its core idea is
PagedAttention, which manages the key-value (KV) cache in non-contiguous pages like virtual
memory, drastically reducing fragmentation. Combined with continuous batching, where new
requests join the running batch as soon as slots free up, vLLM keeps the GPU saturated and
delivers high aggregate throughput for many concurrent users. It exposes an OpenAI-compatible
HTTP API and supports quantized weights, tensor parallelism, and speculative decoding.
