# Mixture of Experts (MoE)

A Mixture-of-Experts layer replaces a single feed-forward network with many expert FFNs and a
router that sends each token to only a few of them (top-k). This decouples total parameters from
per-token compute: a model can hold tens of billions of weights yet activate only a small
fraction per token, so it is cheap to run relative to its size. Hybrid MoE models add a small
number of full-attention layers among cheaper short-convolution or sparse layers. A practical
consequence is a tiny KV cache footprint, which lets a single GPU serve many concurrent requests.
