# mlsys-notes

Learning notes and experiments for understanding modern Machine Learning System.

Currently focusing on LLM serving system and inference optimization.

## Notes
- [Introduction to LLM Inference Part 1](notes/llm-inference-intro-p1.md)
- [ORCA paper review](notes/orca.md)
- [PagedAttention paper review](notes/pagedattention.md)
- [tinyorca deep dive](notes/tinyorca.md)

### Distributed
- [Sarathi-Serve paper review](notes/sarathi-serve.md)
- [How Multiprocess Serving Works in vLLM](notes/how-mp-serving-works-in-vllm.md)
- [Tensor Parallelism](notes/distributed/tp.md)

### Diffusion
- [Accelerating Diffusion Inference via Caching](notes/accelerating-diffusion-inference-via-caching.md)

## Labs
- [microengine](labs/microengine/README.md): a minimal serving engine
- [tinyorca](https://github.com/junuxyz/tinyorca): a minimal implementation of [ORCA](https://www.usenix.org/system/files/osdi22-yu.pdf)
