# SGLang

SGLang is an inference engine and frontend language for programs that call LLMs repeatedly.
Its signature feature is RadixAttention, which stores shared prompt prefixes in a radix tree so
the KV cache is reused across requests that share context, such as few-shot prompts or
multi-turn chats. This makes structured generation, agents, and tool-use loops much cheaper than
recomputing the prefix every call. SGLang targets low latency for complex prompting patterns and
also supports continuous batching and quantization.
