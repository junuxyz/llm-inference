_This is **Part 1** of LLM Inference and Inference Optimization From First Principles. It covers the basic performance model behind LLM inference and may end up either as a standalone post or as part of a longer article._

## how LLM inference works

### what is LLM inference?

LLM inference simply means generating output tokens from input tokens (the _prompt_) using a trained model.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-1.png" width="540" />
  <br />
  <sub>Figure 1. LLM inference generates output tokens autoregressively from an input prompt.<sup><a href="#reference-1">[1]</a></sup></sub>
</p>

### why LLM inference is becoming more and more important

Training is a large upfront expense. More realistically, it is a sequence of large expenses spread across multiple runs, experiments, and model iterations. But even so, training is still bounded in a way inference is not. A training run eventually ends. Once the weights are fixed and the model is deployed, that particular training bill stops growing.

Inference works differently. It is not a one-time expense but an operating cost that repeats every time someone uses the model. Every prompt has to be processed, every output token has to be generated, and every one of those requests consumes accelerator time. If nobody uses the model, the serving cost is close to zero. But if the model becomes widely useful, inference cost scales with that success.<sup><a href="#reference-2">[2]</a></sup>

That is what makes inference economics so important. A model can be expensive to train and still make sense if it is trained once and then reused many times. But a model that is expensive to serve becomes more costly with every additional user, with every longer context window, and with every generated token. At product scale, that recurring serving bill can become comparable to the original training cost or even exceed it.<sup><a href="#reference-2">[2]</a></sup>

That is why people spend so much time optimizing inference. Once a model is in production, the central question is no longer just whether the model is good. It is whether the model can be served fast enough and cheaply enough for real usage to make economic sense.

> **The future is overwhelmingly inference.**<sup><a href="#reference-3">[3]</a></sup>

## what inference engine optimizes

An inference engine is usually trying to do two things at once:
1. lower the cost per generated token
2. reduce the latency of generation

On the cost side, the key question is how much it costs to generate each token. While serving has several cost components, GPU time (or more broadly accelerator time) is usually the dominant one. This article uses the NVIDIA H100 as a reference because it is the GPU most readers will already have in mind for large-model serving.

This is effectively a GPU-cost-per-hour problem. Whether you buy GPUs or rent them from a cloud provider, the hourly bill is mostly fixed. What changes is how much useful generation you get from that fixed spend. An underutilized GPU costs as much as a busy one, which is why utilization matters so much.

Latency matters for a different reason. It directly affects user experience. For offline workloads such as large-scale summarization or synthetic-data generation, this matters less. But for interactive systems such as chat apps and coding agents, latency matters just as much.

The central distinction for reasoning about both cost and latency is the separation between prefill and decode.

### two phases in LLM inference: prefill and decode

Modern LLMs are typically built as decoder-only transformers<sup><a href="#reference-4">[4]</a></sup>. These models generate text autoregressively. Given a sequence of tokens, the model predicts the next token and so on.

Generation largely proceeds in two stages.

First, the model processes the entire prompt to produce the first output token. This stage is called prefill. After prefill, the model generates tokens one at a time, appending each new token to the sequence and feeding it back into the model. This iterative stage is called decode. So for any request, there is one prefill and multiple decode steps.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-2.png" width="540" />
  <br />
  <sub>Figure 2. One prefill pass processes the full prompt, followed by repeated decode steps that generate one token at a time.<sup><a href="#reference-5">[5]</a></sup></sub>
</p>

This difference comes from the dependency structure.

During prefill, the whole prompt is already known. Within each transformer layer, the model can process all prompt tokens together under a causal mask, so computation can be parallelized across the sequence dimension. In fact, this token-parallel computation is one of the key advantages of the Transformer architecture.<sup><a href="#reference-6">[6]</a></sup>

During decode, each new token must be generated before the model can compute the next one. This is because token $t+1$ depends on the token generated at step $t$. As a result, generation becomes sequential across time. The model can still parallelize computation across layers, heads, and batch elements, but the sequence dimension can no longer be parallelized in the same way.

These two phases are crucial for understanding the GPU utilization problem and the performance model developed next.

## the GPU utilization problem

Modern GPUs such as the NVIDIA H100 are extremely capable. On paper, the H100 SXM offers about $3.35\,\text{TB/s}$ of HBM bandwidth and roughly $\sim 1000\,\text{TFLOP/s}$ dense BF16 peak. On paper that should make LLM inference very fast.

However, in practice, LLM inference often fails to use the GPU's full compute capacity. Even when running large models, the GPU spends much of its time waiting for data rather than doing arithmetic. This inefficient utilization raises the cost per generated token, which is exactly what we want to avoid.

To understand why this happens, we need a performance model that captures the relationship between compute work and memory movement. One of the most widely used models for this purpose is the roofline model.

### roofline model

At a high level, any computer program does some mix of two things:
1. moving bytes toward the compute unit
2. doing arithmetic on those bytes

From a performance standpoint, runtime often comes down to which of those two costs dominates.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-3.png" width="540" />
  <br />
  <sub>Figure 3. The roofline model separates memory-bound and compute-bound regimes as arithmetic intensity increases.<sup><a href="#reference-7">[7]</a></sup></sub>
</p>

The roofline model<sup><a href="#reference-8">[8]</a></sup> turns that intuition into a hardware limit using two quantities: peak compute throughput and peak memory bandwidth. Its x-axis is arithmetic intensity. Its y-axis is performance, usually measured in FLOP/s.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-4.png" width="540" />
  <br />
  <sub>Figure 4. A literal roof image to build intuition for why the roofline model has a sloped section and a flat ceiling.<sup><a href="#reference-9">[9]</a></sup></sub>
</p>

Before the turning point where the line flattens, performance is limited by memory bandwidth. After that point, it is limited by compute throughput. The shape looks like a roof because modern chips can add arithmetic capacity more easily than memory bandwidth; it is much easier to add more math than to feed that math with data.


<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-5.png" width="480" />
  <br />
  <sub>Figure 5. NVIDIA H100 SXM model-card numbers highlighting the gap between dense BF16 compute throughput and HBM bandwidth; the * refers to sparsity-assisted peak throughput.</sub>
</p>

NVIDIA lists the H100 SXM at about $\sim 1{,}000\,\text{TFLOP/s}$ of BF16 Tensor Core throughput (without sparsity) and $3.35\,\text{TB/s}$ of memory bandwidth. In other words, the chip can do arithmetic far faster than it can pull bytes from memory. That gap has widened over time. This is the basic idea behind the _memory wall_: you need higher arithmetic intensity to reach the compute roof.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-6.png" width="540" />
  <br />
  <sub>Figure 6. The memory wall: compute throughput has scaled faster than memory bandwidth, making data movement the bottleneck more often.<sup><a href="#reference-10">[10]</a></sup></sub>
</p>

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-7.png" width="540" />
  <br />
  <sub>Figure 7. A factory-and-supply-chain metaphor for compute vs. memory: fast machines are useless if the bridge delivering data is too narrow.<sup><a href="#reference-11">[11]</a></sup></sub>
</p>

One way to picture this is to think of compute as a factory and memory bandwidth as the bridge delivering raw materials. The machines can run much faster than the bridge can deliver new material, and that mismatch is what the roofline model captures.

> **Note**
> Here, "compute" means how many operations the compute unit can perform. In the context of AI workloads on GPUs, we usually measure it in FLOP/s (floating-point operations per second) or in FLOPs (total floating-point operations). "Memory," or more precisely memory bandwidth, is the speed at which data can be moved around. In LLM inference, that often means moving data from HBM into faster on-chip memory such as shared memory or registers.

### arithmetic intensity

We have been referring to the x-axis without defining it precisely. That quantity is arithmetic intensity. It is the main lever systems people try to improve.

Arithmetic intensity is the ratio between compute work and memory traffic:

$$\text{Arithmetic Intensity} = \frac{\text{Operations}}{\text{Bytes Moved}}$$

If there are more operations done on the data than bytes moved, we get higher arithmetic intensity. Conversely, if we move lots of bytes but do little work on them, we end up with low arithmetic intensity.

For example, suppose two programs both load the same two numbers, $x$ and $y$, from memory.

- Program A computes only $x + y$. That is roughly one arithmetic operation on the same input bytes.
- Program B computes something like $(x + y \cdot y) / x + y$. That is several arithmetic operations on the same input bytes.

Both programs moved the same inputs, but Program B did more work on them before finishing. So Program B has higher arithmetic intensity. In AI workloads, those operations are usually floating-point operations, so we typically measure compute in FLOPs.

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Moved}}$$
### adding it all up


<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-8.png" width="540" />
  <br />
  <sub>Figure 8. Idealized H100 roofline showing the ridge point where performance transitions from bandwidth-limited to compute-limited.</sub>
</p>

> **Note**
> Crossing the ridge point does not mean the workload will automatically run near peak compute throughput. It means memory bandwidth is less likely to be the main bottleneck, so compute-side limits become more important.

When arithmetic intensity is low, performance sits on the bandwidth-limited slope. Only after each byte is reused enough times does the bottleneck shift to the compute units, at which point performance flattens at the compute ceiling.

> Since compute on modern hardware, including GPUs, is much faster than memory movement, **you will not reach peak performance unless your workload has high arithmetic intensity.**

In other words, a useful systems goal is to increase arithmetic intensity so the hardware spends more time doing math and less time waiting on bytes. So arithmetic intensity gives us a practical way to think about LLM inference.

## revisiting prefill vs decode through roofline analysis

The roofline model is a good way to think about the different characteristics of prefill and decode. Both phases run the same transformer layers with the same weights. What changes is the balance between compute and memory traffic, and that is what changes arithmetic intensity.

### during prefill phase

During prefill, the model processes the whole prompt at once. If the prompt length is $T$, each transformer layer runs over all $T$ tokens together.

The dominant operations are large batched matrix-matrix multiplications, so prefill usually keeps the compute units much busier than decode. Every prompt token activates the full attention and MLP stack.

On the memory side, the main terms are model-weight traffic, KV-cache writes, and temporary activations<sup><a href="#reference-12">[12]</a></sup>.

A $32\text{B}$-parameter model in bf16<sup><a href="#reference-13">[13]</a></sup>, for example, stores about $32\text{B} \times 2\ \text{bytes} = 64\,\text{GB}$ of weights. So we need to stream the whole model's parameters from HBM to the compute units.

There is also activation memory, which is the temporary tensors created during the forward pass. These are mainly hidden states, Q/K/V projections, attention outputs, and MLP intermediate tensors. It is hard to calculate the exact usage of this from first principles because it depends on multiple configurations (batch size, sequence length, kernels, and the runtime). In practice, people often use broad heuristics (e.g. $10\text{--}30\%$ of model weight<sup><a href="#reference-14">[14]</a></sup>) for capacity planning. We will talk more about that in the memory capacity section.

Prefill also writes something called KV cache. KV cache is the stored K and V values for each token at each transformer layer. We write it during prefill so that decode does not need to recompute the K and V values of older tokens.

KV-cache write does add memory traffic, but it is mostly a one-time write over the prompt. Also compared with the amount of batched compute happening during prefill, it is usually a much smaller term.

So prefill ends up with relatively high arithmetic intensity. Even though it needs to load model weights and write KV states for all tokens, it performs a large amount of computation over the entire prompt.

### during decode phase

Unlike prefill, decode processes one token at a time. The model still runs the same layer stack, but now the major projections and MLPs operate on something closer to `[1 × hidden]` than `[T × hidden]`, thanks to KV cache. This is much closer to matrix-vector multiplication, which is known to have low arithmetic intensity.

However, memory traffic does not shrink the same way. The GPU still needs to move the same model weights just like it did for prefill, but now those weights are loaded on every decode step.

Activation memory is smaller compared to prefill because only one new token is being processed at a time, though it is still not literally zero.

We also need to read the KV cache of all previous tokens in the sequence during every decode step. As the sequence grows, the KV read grows with it.

So while the compute per decode step is small, the bytes moved per step stay large and often get worse as context size grows. That is why decode ends up with lower arithmetic intensity and is much more likely to be memory-bound.

### putting it together

> During prefill, large batched GEMMs give the model many FLOPs for each byte it moves. During decode, the model processes only one token at a time, rereads the layer weights from HBM, and repeatedly reads the growing KV cache. That sharply lowers FLOPs per byte.

That is why prefill and decode land in different parts of the roofline model.

Now let’s go through a concrete example with Qwen3-32B on a single H100 SXM.<sup><a href="#reference-15">[15]</a></sup>

## example: Qwen3-32B on H100 SXM

To make the discussion concrete, consider serving a single request with Qwen3-32B on one NVIDIA H100 SXM. We assume `bf16` inference, $64$ transformer layers, hidden size $d = 5120$, MLP size $d_{ff} = 25600$, $64$ query heads, $8$ KV heads, head dimension $128$, and an H100 roofline of about $3.35\,\text{TB/s}$ HBM bandwidth and $\sim 1000\,\text{TFLOP/s}$ dense BF16 peak.

For prefill, the important qualitative result is that the workload lands on the compute-favored side of the roofline. Using the same Qwen3-32B assumptions as above, a $1000$-token prompt comes out to about $63.46\,\text{TFLOPs}$ of core transformer work and an arithmetic intensity of about $1012.6\ \text{FLOPs/byte}$, which is comfortably above the H100 ridge point of roughly $298.5\ \text{FLOPs/byte}$. The exact number will shift with accounting choices, but the conclusion is robust: prefill has enough batched matrix work to look compute-favored rather than bandwidth-limited.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-9.png" width="540" />
  <br />
  <sub>Figure 9. Prefill-phase arithmetic intensity on H100, showing that the example workload lies on the compute-favored side of the roofline.</sub>
</p>

For decode, the picture flips. One decode step still streams essentially the same layer weights, but now it does only one token's worth of dense projection and MLP work while also rereading the growing KV cache. At context length $L = 1000$, the same first-order model gives an arithmetic intensity of about $1.03\ \text{FLOPs/byte}$, and even as $L$ grows very large the asymptote is only about $8\ \text{FLOPs/byte}$. So decode remains firmly memory-bound, and that qualitative conclusion is also robust to accounting detail.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-10.png" width="540" />
  <br />
  <sub>Figure 10. Decode-phase arithmetic intensity on H100, illustrating why single-token decode remains far below the ridge point and stays memory-bound.</sub>
</p>

Appendix A provides the full derivation, including smaller per-layer operations such as RMSNorm, RoPE, softmax, residual adds, and gating terms. The appendix is still a first-order model rather than a runtime simulator, but it is the right place for the detailed FLOPs and bytes bookkeeping.

## performance metrics

So far, the discussion has focused on the internal structure of inference and the hardware limits that shape it. In real systems, those effects show up through a smaller set of serving metrics.

### latency and throughput

**Latency** is the time between when a user's request is sent and when the generation finishes. A useful split mirrors the two inference phases. TTFT maps to prefill. TPOT maps to decode.

**TTFT** (Time-To-First-Token)

TTFT measures the time between receiving a user's request and generating the first output token, usually in seconds. It roughly captures the prefill-dominated part of latency, though in real systems it also includes queueing and runtime overhead. Lower TTFT means the stream starts sooner, which matters a lot in latency-sensitive applications.

**TPOT** (Time-Per-Output-Token)

TPOT measures the average latency between output tokens after the first one. It is often discussed as average ITL, or inter-token latency, during decode. It roughly tracks decode speed, though sampling and scheduler overhead also contribute. Lower TPOT means the request streams faster and finishes sooner.

> **Latency ≈ TTFT + (TPOT × (output tokens - 1))**

This is still an approximation, because per-token latency is not perfectly constant in real systems.

**Throughput** is the number of tokens or requests the system can process per second. Higher throughput means the system handles larger workloads more efficiently.

The core metrics here are RPS, TPS, and goodput.

**RPS** (Requests-Per-Second)

RPS measures how many requests the model can successfully process per second.

**TPS** (Tokens-Per-Second)

TPS measures how many tokens the model can successfully process per second. RPS depends heavily on workload shape, such as whether requests are short or long. TPS is often a more convenient unit, but it is still workload-dependent. When people report TPS, it is worth checking whether they mean input tokens, output tokens, or both.

**Goodput**

Goodput measures the rate of requests or tokens that meet the target SLO, or service-level objective.

For example, suppose a system serves $100$ requests per second, but only $70$ of them satisfy your TTFT and TPOT targets. The raw throughput is $100\,\text{RPS}$, but the goodput under that SLO is $70\,\text{RPS}$.

Goodput differs from throughput because it includes latency-related constraints rather than just raw volume.

> **Note**
> This article does not cover every metric in detail, for example TTLT, because the goal is to build a clear mental model for LLM engines and InferenceOps. The metrics above are enough for that purpose. For a broader survey of inference metrics, see this overview<sup><a href="#reference-17">[17]</a></sup>.

This gives us the basic mapping. Prefill largely determines TTFT. Decode largely determines TPOT. Throughput depends on how efficiently the hardware is used across both phases. The next section shifts from bandwidth-limited performance to memory capacity.

## how much memory is needed to decode?

So far, the discussion has treated memory mainly as a bandwidth problem. The question was how many bytes must move through HBM during a forward pass. This section asks a different question. How many bytes must fit in VRAM at all?

This distinction matters.

> **Note**
> In the roofline discussion above, weights were mostly a **traffic** question. Here they are a **capacity** question. The question is how many bytes must live in VRAM at all.

A simple way to see that distinction is to visualize one decode request on an H100 SXM with $80\,\text{GB}$ of VRAM.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-11.png" width="540" />
  <br />
  <sub>Figure 11. An empty H100 GPU memory budget with 80 GB of VRAM before allocating model weights, KV cache, or activations.</sub>
</p>

During inference, model weights must stay resident in VRAM. Activations and KV cache add pressure on top of that base footprint.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-12.png" width="540" />
  <br />
  <sub>Figure 12. After loading the model, GPU memory is already mostly occupied by resident weights, with only a small activation footprint shown separately.</sub>
</p>

While we can use various dtypes for model weights, `bf16` is still a common baseline. At a rough back-of-the-envelope level, a $32\text{B}$-parameter model in `bf16` needs about $32\text{B} \times 2\ \text{bytes} = 64\,\text{GB}$ of raw weight storage. In this section, treat memory sizes as rough decimal GB estimates rather than binary GiB accounting.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-13.png" width="540" />
  <br />
  <sub>Figure 13. For Qwen3-32B in bf16, model weights alone take roughly 64 GB, illustrating why weight residency dominates VRAM capacity.</sub>
</p>

Activation memory is harder to estimate from first principles because it depends on batch size, sequence length, kernels, and runtime behavior. In practice, rough heuristics such as $10\text{--}30\%$ of model weight are sometimes used for planning. Simple decode usually has a much smaller live activation footprint.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-14.png" width="540" />
  <br />
  <sub>Figure 14. Activation memory is typically much smaller than model weights in decode, though it grows with larger prefills and larger batches.</sub>
</p>

After the model is loaded, most of the $80\,\text{GB}$ budget is already occupied. The remaining headroom goes to KV cache, activation and workspace buffers, and runtime overhead. In practice, usable headroom is smaller than the naive $80\,\text{GB} - 64\,\text{GB}$ because CUDA and the serving stack also consume memory.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-15.png" width="540" />
  <br />
  <sub>Figure 15. Remaining VRAM headroom after loading about 64 GB of model weights: roughly 16 GB for KV cache, activations, workspace buffers, and runtime overhead.</sub>
</p>

### serving one request

Consider one request in its decode phase with a KV cache of $2{,}048$ tokens. At that point, GPU memory already contains the KV cache of the prefilled prompt plus the tokens decoded so far.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-16.png" width="540" />
  <br />
  <sub>Figure 16. Memory layout for a single decode request, with one request's KV cache added beside the model weights and example KV sizes shown for different context lengths.</sub>
</p>

Recalling how we calculate KV cache, Qwen3-32B uses GQA, so the KV width is $8 \times 128 = 1024$.

For $2{,}048$ tokens of KV cache, the total becomes

$$
2 \;(\text{for K and V})
\times 64 \;(\text{layers})
\times 1024 \;(\text{KV width})
\times 2 \;(\text{bytes for bf16})
\times 2{,}048 \;(\text{token length})
= 536{,}870{,}912 \text{ bytes}
\approx 0.54 \text{ GB}.
$$

At around a $2\text{K}$ context, the memory story is still dominated by model weights, not KV cache.

### why batching requests makes each token generation cheaper

A single decode request generates only one token per step and does not fully use the remaining memory headroom. That is what batching exploits.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-17.png" width="540" />
  <br />
  <sub>Figure 17. With only one active request, each decode step still produces just one new token, leaving substantial VRAM headroom underutilized.</sub>
</p>

If multiple requests are active, the remaining VRAM can hold additional KV cache and activation or workspace state. That lets more requests run together on the same GPU. Figure 18 shows four requests for illustration, but the real batch is limited by available VRAM and latency targets.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-18.png" width="540" />
  <br />
  <sub>Figure 18. Example batched serving layout with four requests, each carrying roughly a 2K-token KV cache, while activation memory still remains relatively small and ephemeral.</sub>
</p>

Activation memory grows with the amount of work in the current forward pass, so it increases with sequence length and batch size. In some cases its share rises. In many practical settings, though, model weights and KV cache still dominate long-lived memory use. Unlike KV cache, activation memory is not accumulated across decode steps. It exists only for that forward pass.

<p align="center">
  <img src="../assets/notes/llm-inference-intro-p1/llm-inference-intro-p1-19.png" width="540" />
  <br />
  <sub>Figure 19. Batched decode turns one forward pass into four output tokens, improving weight reuse and throughput across simultaneously served requests.</sub>
</p>

This makes token generation cheaper because one batched decode step can generate one token for each active request. That amortizes heavy model-weight traffic across more useful work. Intuitively, the workload becomes less matrix-vector-like and more matrix-matrix-like.

We will look at more advanced inference optimization techniques in Part 2.

## summary

The most important takeaways are:

- LLM inference has two distinct phases: prefill and decode.
- Prefill processes the full prompt in parallel, so it usually has much higher arithmetic intensity.
- Decode generates one token at a time, so it is much more likely to be memory-bandwidth-bound.
- The roofline model is a useful way to reason about why these two phases behave so differently on modern GPUs.
- In serving, model weights dominate both bandwidth traffic and VRAM capacity, which is why weight reuse matters so much.
- Batching improves throughput because one forward pass can serve multiple requests and amortize weight movement across more useful work.

At that point, the high-level split is simple. Prefill is mainly a compute problem. Decode is mainly a memory problem. Good inference systems are built around that split. Part 2 builds on that foundation and moves into the more advanced techniques modern inference engines use in practice.

## appendix a. Qwen3-32B on H100 SXM accounting

This appendix collects the first-order accounting that the main text skips. It is still not a cycle-accurate runtime model. The goal is to keep the dominant matrix terms, add the smaller per-layer terms omitted in the body, and show that the main roofline conclusion still holds.

Throughout Appendix A, use the Qwen3-32B configuration from the example.

- $64$ transformer layers
- hidden size $d = 5120$
- MLP size $d_{ff} = 25600$
- $64$ query heads
- $8$ KV heads
- head dimension $128$
- query width $d_q = 64 \times 128 = 8192$
- KV width $d_{kv} = 8 \times 128 = 1024$
- vocabulary size $151{,}936$
- `bf16` $= 2$ bytes per element

### per-layer operations

At a high level, one transformer block does:

input  
$\to$ RMSNorm  
$\to$ $Q, K, V$ projections  
$\to$ RoPE  
$\to$ attention scores  
$\to$ softmax  
$\to$ attention-weighted value matmul  
$\to$ output projection  
$\to$ residual add  
$\to$ RMSNorm  
$\to$ gated MLP  
$\to$ residual add

We will count FLOPs by following that execution order directly. The dense matrix multiplies dominate. This appendix also includes the smaller per-layer terms so the bookkeeping is more complete.

> **Note**
> A matrix multiply of an $m \times k$ matrix with a $k \times n$ matrix costs about $2mkn$ FLOPs. That is the rule used throughout this appendix.

Qwen3-32B also uses GQA (grouped-query attention)<sup><a href="#reference-16">[16]</a></sup>, so the query side and KV side are intentionally asymmetric:

- $Q$: $[S \times 5120] \cdot [5120 \times 8192]$
- $K$: $[S \times 5120] \cdot [5120 \times 1024]$
- $V$: $[S \times 5120] \cdot [5120 \times 1024]$
- $O$: $[S \times 8192] \cdot [8192 \times 5120]$

That asymmetry reduces both compute and KV-cache size on the KV side.

### prefill FLOPs

During prefill, the model processes a prompt of length $T$ in one pass.

The embedding lookup itself is mostly a memory-access operation, so we treat it as negligible FLOPs and start from the first transformer block.

#### per-layer prefill FLOPs, step by step

For one layer, let the input hidden state have shape $[T \times d] = [T \times 5120]$.

**1. first RMSNorm**

RMSNorm is an elementwise operation over the hidden width. A rough first-order count is a small constant number of operations per element, which we approximate as $4$ FLOPs.

$$
F_{\text{RMSNorm}_1}(T) \approx 4Td = 20{,}480\,T
$$

**2. query projection**

Shapes:

- input: $[T \times 5120]$
- weight: $[5120 \times 8192]$
- output: $[T \times 8192]$

$$
F_Q = 2T \cdot 5120 \cdot 8192 = 83{,}886{,}080\,T
$$

**3. key projection**

Shapes:

- input: $[T \times 5120]$
- weight: $[5120 \times 1024]$
- output: $[T \times 1024]$

$$
F_K = 2T \cdot 5120 \cdot 1024 = 10{,}485{,}760\,T
$$

**4. value projection**

Shapes:

- input: $[T \times 5120]$
- weight: $[5120 \times 1024]$
- output: $[T \times 1024]$

$$
F_V = 2T \cdot 5120 \cdot 1024 = 10{,}485{,}760\,T
$$

**5. RoPE on queries and keys**

RoPE is another elementwise operation. A common rough count is about $3$ FLOPs per rotated element. Applying it to both $Q$ and $K$ gives:

$$
F_{\text{RoPE}}(T) \approx 3T(d_q + d_{kv}) = 27{,}648\,T
$$

**6. attention scores $QK^\top$**

During prefill, each head attends over the full prompt. For one head, the score matrix is $[T \times 128] \cdot [128 \times T] \to [T \times T]$, so the cost is about $2T^2 \cdot 128$ FLOPs. Under a causal mask, only the lower triangle is used, so a first-order count is:

$$
F_{QK^\top}(T) = 64 \cdot 2 \cdot 128 \cdot \frac{T(T+1)}{2} = 8192\,T(T+1)
$$

**7. softmax over attention scores**

Softmax is harder to count exactly because the true cost depends on how exp, reduction, and division are implemented. For a simple first-order model, we approximate it as $5$ FLOPs per score element:

$$
F_{\text{softmax}}(T) \approx 5 \cdot 64 \cdot \frac{T(T+1)}{2} = 160\,T(T+1)
$$

**8. attention output $(QK^\top)V$**

For one head, this is $[T \times T] \cdot [T \times 128] \to [T \times 128]$, again under a causal mask:

$$
F_{\text{attn} \times V}(T) = 64 \cdot 2 \cdot 128 \cdot \frac{T(T+1)}{2} = 8192\,T(T+1)
$$

**9. output projection**

Shapes:

- input: $[T \times 8192]$
- weight: $[8192 \times 5120]$
- output: $[T \times 5120]$

$$
F_O = 2T \cdot 8192 \cdot 5120 = 83{,}886{,}080\,T
$$

**10. first residual add**

Adding the attention output back to the residual stream is one addition per hidden element:

$$
F_{\text{residual}_1}(T) \approx Td = 5120\,T
$$

**11. second RMSNorm**

$$
F_{\text{RMSNorm}_2}(T) \approx 4Td = 20{,}480\,T
$$

**12. MLP gate projection**

Shapes:

- input: $[T \times 5120]$
- weight: $[5120 \times 25600]$
- output: $[T \times 25600]$

$$
F_{\text{gate proj}}(T) = 2T \cdot 5120 \cdot 25600 = 262{,}144{,}000\,T
$$

**13. MLP up projection**

Shapes:

- input: $[T \times 5120]$
- weight: $[5120 \times 25600]$
- output: $[T \times 25600]$

$$
F_{\text{up proj}}(T) = 2T \cdot 5120 \cdot 25600 = 262{,}144{,}000\,T
$$

**14. gating elementwise operation**

For the gated activation, we use a rough $6$-FLOP estimate per intermediate element:

$$
F_{\text{gating}}(T) \approx 6Td_{ff} = 153{,}600\,T
$$

**15. MLP down projection**

Shapes:

- input: $[T \times 25600]$
- weight: $[25600 \times 5120]$
- output: $[T \times 5120]$

$$
F_{\text{down proj}}(T) = 2T \cdot 25600 \cdot 5120 = 262{,}144{,}000\,T
$$

**16. second residual add**

$$
F_{\text{residual}_2}(T) \approx Td = 5120\,T
$$

Collecting those terms:

$$
F_{\text{attn proj}}(T) = F_Q + F_K + F_V + F_O = 188{,}743{,}680\,T
$$

$$
F_{\text{MLP}}(T)
= F_{\text{gate proj}}(T) + F_{\text{up proj}}(T) + F_{\text{down proj}}(T)
= 786{,}432{,}000\,T
$$

$$
F_{\text{attn matmuls}}(T) = F_{QK^\top}(T) + F_{\text{attn} \times V}(T) = 16{,}384\,T(T+1)
$$

$$
F_{\text{RMSNorms}}(T) = F_{\text{RMSNorm}_1}(T) + F_{\text{RMSNorm}_2}(T) \approx 40{,}960\,T
$$

$$
F_{\text{residual adds}}(T) = F_{\text{residual}_1}(T) + F_{\text{residual}_2}(T) \approx 10{,}240\,T
$$

So the full first-order prefill total for one layer is:

$$
F_{\text{layer, prefill}}(T)
\approx
975{,}408{,}128\,T + 16{,}544\,T(T+1)
$$

Across all $64$ layers:

$$
F_{\text{model, prefill}}(T)
\approx
64 \cdot F_{\text{layer, prefill}}(T)
= 62{,}426{,}120{,}192\,T + 1{,}058{,}816\,T(T+1)
$$

At $T = 1000$:

$$
F_{\text{model, prefill}}(1000)
\approx
63{,}485{,}995{,}008{,}000
$$

So even after adding the smaller terms, prefill is still about $63.49\,\text{TFLOPs}$ for a $1000$-token prompt, which is only slightly above the dominant-terms-only estimate.

### bytes moved during prefill

For a roofline-style estimate, the main memory terms during prefill are:

- model-weight traffic
- KV-cache write
- an optional activation/workspace term that is explicitly approximate

The dominant repeated weights per layer are the four attention projections plus the three MLP projections:

$$
5120 \cdot 8192 + 5120 \cdot 1024 + 5120 \cdot 1024 + 8192 \cdot 5120
= 94{,}371{,}840
$$

$$
5120 \cdot 25600 + 5120 \cdot 25600 + 25600 \cdot 5120
= 393{,}216{,}000
$$

So the dominant repeated block weights total:

$$
487{,}587{,}840 \text{ parameters per layer}
$$

Across $64$ layers, that is:

$$
31{,}205{,}621{,}760 \text{ parameters}
$$

In bf16:

$$
M_{\text{weights}} \approx 62{,}411{,}243{,}520 \text{ bytes} \approx 62.41 \text{ GB}
$$

The smaller block parameters, such as the two RMSNorm weight vectors per layer, are tiny by comparison:

$$
64 \cdot 2 \cdot 5120 \cdot 2 = 1{,}310{,}720 \text{ bytes}
$$

so they do not materially change the memory picture.

For KV-cache write, each new token stores one $K$ vector and one $V$ vector of width $1024$ at each of the $64$ layers:

$$
M_{\text{KV write per token}}
= 2 \cdot 64 \cdot 1024 \cdot 2
= 262{,}144 \text{ bytes}
\approx 0.000262 \text{ GB}
$$

So for a prompt of length $T$:

$$
M_{\text{KV write}}(T) \approx 0.000262 \text{ GB} \cdot T
$$

An optional activation/workspace term is harder to derive cleanly from the architecture alone because it depends on kernel fusion, tiling, scheduler behavior, and whether intermediates spill to HBM. For a first-order model, it is best left as an explicitly approximate additive term:

$$
M_{\text{act/ws, prefill}}(T) \approx \delta_{\text{prefill}}(T)
$$

where $\delta_{\text{prefill}}(T)$ is implementation-dependent and is usually smaller than the dominant model-weight traffic in this simplified roofline view.

Putting those together:

$$
M_{\text{prefill}}(T)
\approx
62.41 \text{ GB} + 0.000262 \text{ GB} \cdot T + \delta_{\text{prefill}}(T)
$$

### prefill arithmetic intensity

Ignoring the optional activation/workspace term for the moment, the more complete prefill arithmetic-intensity expression is:

$$
AI_{\text{prefill}}(T)
\approx
\frac{62{,}426{,}120{,}192\,T + 1{,}058{,}816\,T(T+1)}
{62{,}411{,}243{,}520 + 262{,}144\,T}
\quad \text{FLOPs/byte}
$$

At $T = 1000$:

$$
AI_{\text{prefill}}(1000) \approx 1012.9 \ \text{FLOPs/byte}
$$

Even after including the smaller per-layer terms, prefill stays comfortably on the compute-favored side of the H100 roofline.

### decode FLOPs

During decode, we generate one new token while attending over an existing context of length $L$.

The execution order is the same as in prefill, but now each dense layer is applied to one new token while attention reads across the existing KV cache.

#### per-layer decode FLOPs, step by step

For one layer, let the new-token hidden state have shape $[1 \times d] = [1 \times 5120]$.

**1. first RMSNorm**

$$
F_{\text{RMSNorm}_1,\text{decode}} \approx 4d = 20{,}480
$$

**2. query projection**

$$
F_{Q,\text{decode}} = 2 \cdot 5120 \cdot 8192 = 83{,}886{,}080
$$

**3. key projection**

$$
F_{K,\text{decode}} = 2 \cdot 5120 \cdot 1024 = 10{,}485{,}760
$$

**4. value projection**

$$
F_{V,\text{decode}} = 2 \cdot 5120 \cdot 1024 = 10{,}485{,}760
$$

**5. RoPE on the new query and key**

$$
F_{\text{RoPE, decode}} \approx 3(d_q + d_{kv}) = 27{,}648
$$

**6. attention scores $QK^\top$**

Now the new query attends over $L$ previous positions. For each head this is $[1 \times 128] \cdot [128 \times L] \to [1 \times L]$:

$$
F_{QK^\top,\text{decode}}(L) = 64 \cdot 2 \cdot 128 \cdot L = 16{,}384\,L
$$

**7. softmax over the $L$ scores per head**

$$
F_{\text{softmax, decode}}(L) \approx 5 \cdot 64 \cdot L = 320\,L
$$

**8. attention output $(QK^\top)V$**

$$
F_{\text{attn} \times V,\text{decode}}(L) = 16{,}384\,L
$$

**9. output projection**

$$
F_{O,\text{decode}} = 2 \cdot 8192 \cdot 5120 = 83{,}886{,}080
$$

**10. first residual add**

$$
F_{\text{residual}_1,\text{decode}} \approx d = 5120
$$

**11. second RMSNorm**

$$
F_{\text{RMSNorm}_2,\text{decode}} \approx 4d = 20{,}480
$$

**12. MLP gate projection**

$$
F_{\text{gate proj, decode}} = 2 \cdot 5120 \cdot 25600 = 262{,}144{,}000
$$

**13. MLP up projection**

$$
F_{\text{up proj, decode}} = 2 \cdot 5120 \cdot 25600 = 262{,}144{,}000
$$

**14. gating elementwise operation**

$$
F_{\text{gating, decode}} \approx 6d_{ff} = 153{,}600
$$

**15. MLP down projection**

$$
F_{\text{down proj, decode}} = 2 \cdot 25600 \cdot 5120 = 262{,}144{,}000
$$

**16. second residual add**

$$
F_{\text{residual}_2,\text{decode}} \approx d = 5120
$$

Collecting the decode terms:

$$
F_{\text{attn proj, decode}} = F_{Q,\text{decode}} + F_{K,\text{decode}} + F_{V,\text{decode}} + F_{O,\text{decode}} = 188{,}743{,}680
$$

$$
F_{\text{MLP, decode}} = F_{\text{gate proj, decode}} + F_{\text{up proj, decode}} + F_{\text{down proj, decode}} = 786{,}432{,}000
$$

$$
F_{\text{attn matmuls, decode}}(L) = F_{QK^\top,\text{decode}}(L) + F_{\text{attn} \times V,\text{decode}}(L) = 32{,}768\,L
$$

$$
F_{\text{RMSNorms, decode}} \approx F_{\text{RMSNorm}_1,\text{decode}} + F_{\text{RMSNorm}_2,\text{decode}} = 40{,}960
$$

$$
F_{\text{residual adds, decode}} \approx F_{\text{residual}_1,\text{decode}} + F_{\text{residual}_2,\text{decode}} = 10{,}240
$$

So the full first-order decode total for one layer is:

$$
F_{\text{layer, decode}}(L)
\approx
975{,}408{,}128 + 33{,}088\,L
$$

Across all $64$ layers:

$$
F_{\text{model, decode}}(L)
\approx
62{,}426{,}120{,}192 + 2{,}117{,}632\,L
$$

At $L = 1000$:

$$
F_{\text{model, decode}}(1000)
\approx
62{,}428{,}237{,}824
$$

The decode FLOP count hardly changes when we add the smaller ops, because the dense projections and MLP still dominate the arithmetic.

As a separate end-of-model term, the final logits projection for one token would be:

$$
F_{\text{lm head}} \approx 2 \cdot 5120 \cdot 151{,}936 = 1{,}555{,}824{,}640
$$

That term can be significant, but it should be tracked separately because it is not part of the repeated transformer-block total.

### bytes moved during decode

During decode, the main memory terms are:

- model-weight traffic
- KV-cache read
- KV-cache write
- an optional activation/workspace term

The weight term is still approximately:

$$
M_{\text{weights}} \approx 62.41 \text{ GB}
$$

KV-cache read scales with the existing context length:

$$
M_{\text{KV read}}(L) \approx 0.000262 \text{ GB} \cdot L
$$

And each decode step also writes the new token's KV entry:

$$
M_{\text{KV write}} \approx 0.000262 \text{ GB}
$$

As in prefill, there is also some activation and workspace traffic, but it is implementation-dependent and much harder to pin down from first principles:

$$
M_{\text{act/ws, decode}}(L) \approx \delta_{\text{decode}}(L)
$$

So the decode-step memory traffic is:

$$
M_{\text{decode}}(L)
\approx
62.41 \text{ GB} + 0.000262 \text{ GB} \cdot L + 0.000262 \text{ GB} + \delta_{\text{decode}}(L)
$$

As a separate end-of-model term, the lm-head weights are also sizable:

$$
M_{\text{lm head}} \approx 151{,}936 \cdot 5120 \cdot 2 = 1{,}555{,}824{,}640 \text{ bytes} \approx 1.56 \text{ GB}
$$

Again, it is clearer to keep that separate from the repeated transformer-block traffic.

### decode arithmetic intensity

Ignoring the optional activation/workspace term for the moment, the more complete decode arithmetic-intensity expression is:

$$
AI_{\text{decode}}(L)
\approx
\frac{62{,}426{,}120{,}192 + 2{,}117{,}632\,L}
{62{,}411{,}243{,}520 + 262{,}144\,L + 262{,}144}
\quad \text{FLOPs/byte}
$$

At $L = 1000$:

$$
AI_{\text{decode}}(1000) \approx 1.03 \ \text{FLOPs/byte}
$$

And as $L$ becomes very large:

$$
AI_{\text{decode}}(L) \to \frac{2{,}117{,}632}{262{,}144} \approx 8.08 \ \text{FLOPs/byte}
$$

Those numbers are only slightly different from the dominant-terms-only version. That is why the main text does not need to dwell on every small term: once you include RMSNorm, RoPE, softmax, residual adds, and gating terms, the qualitative conclusion does not change. Decode stays firmly memory-bound.


## references

This post was heavily inspired by [LLM Inference Economics From First Principles](https://www.tensoreconomics.com/p/llm-inference-economics-from-first) from Tensor Economics. I found its framing especially useful, and this article is my own attempt to reorganize and explain the ideas in my own way.

<ol>
  <li id="reference-1">Daily Dose of Data, "Regular ML Inference vs. LLM Inference" - used for the opening Figure 1 visual. <a href="https://blog.dailydoseofds.com/p/regular-ml-inference-vs-llm-inference">Link</a></li>
  <li id="reference-2">OpenAI, "AI and efficiency" - especially useful for the point that inference costs scale with usage and can dominate total costs in successful deployed systems. <a href="https://openai.com/index/ai-and-efficiency/">Link</a></li>
  <li id="reference-3">Elon Musk, "The future is overwhelmingly inference." - the short quote used in the motivation section. <a href="https://x.com/elonmusk/status/2029372889959706901">Link</a></li>
  <li id="reference-4">Cameron R. Wolfe, "Decoder-Only Transformers: The Workhorse of Generative LLMs" - a good background explainer for decoder-only transformers. <a href="https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse">Link</a></li>
  <li id="reference-5">Mark Moyou / NVIDIA, "Mastering LLM Inference" - used for the prefill/decode figure. <a href="https://file.notion.so/f/f/bea58367-f949-4c4b-b461-492356378a41/dfb74196-771c-43fa-a1a5-13660d43512a/mark-moyou-nvidia-mastering-llm-inference.pdf?table=block&id=a9061206-58b4-4698-9516-34c4ad9380f1&spaceId=bea58367-f949-4c4b-b461-492356378a41&expirationTimestamp=1773345600000&signature=BFFye7bXp3tBlCdlzG2z_ycDOLXvc0DBPtgnNwkKY2U&downloadName=mark-moyou-nvidia-mastering-llm-inference.pdf">Link</a></li>
  <li id="reference-6">Vaswani et al., "Attention Is All You Need" - the original Transformer paper, mainly for the causal-mask and parallelism discussion. <a href="https://arxiv.org/abs/1706.03762">Link</a></li>
  <li id="reference-7">Roofline-model figure source - where Figure 3 came from. <a href="https://arxiv.org/pdf/2402.16363">Link</a></li>
  <li id="reference-8">Wikipedia, "Roofline model" - a lightweight reference for the roofline concept itself. <a href="https://en.wikipedia.org/wiki/Roofline_model">Link</a></li>
  <li id="reference-9">Pexels, "Orange and Gray Painted Roof Under Cloudy Sky" - used for the literal roof image. <a href="https://www.pexels.com/photo/orange-and-gray-painted-roof-under-cloudy-347152/">Link</a></li>
  <li id="reference-10">Astera Labs, "Breaking Through the Memory Wall" - useful background and the visual source for the memory-wall section. <a href="https://www.asteralabs.com/breaking-through-the-memory-wall/">Link</a></li>
  <li id="reference-11">Horace He, "Brrr... Intro" - source for the compute-vs.-memory factory metaphor figure. <a href="https://horace.io/brrr_intro.html">Link</a></li>
  <li id="reference-12">vLLM Omni documentation, "GPU Memory Utilization" - helpful for thinking about GPU memory terms during serving. <a href="https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/gpu_memory_utilization/">Link</a></li>
  <li id="reference-13">Google Cloud, "BFloat16: The secret to high performance on Cloud TPUs" - background for the bf16 assumptions used in the examples. <a href="https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus?hl=en">Link</a></li>
  <li id="reference-14">vLLM Omni documentation, "Activation Memory" - the source behind the rough activation-memory heuristic. <a href="https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/gpu_memory_utilization/#activation-memory">Link</a></li>
  <li id="reference-15">Hugging Face, "Qwen/Qwen3-32B" - model page for the Qwen3-32B example used in this article. <a href="https://huggingface.co/Qwen/Qwen3-32B">Link</a></li>
  <li id="reference-16">IBM, "What is Grouped-Query Attention?" - a readable explainer for GQA. <a href="https://www.ibm.com/think/topics/grouped-query-attention">Link</a></li>
  <li id="reference-17">BentoML, "LLM Inference Metrics" - a good follow-up for a broader survey of serving metrics. <a href="https://bentoml.com/llm/inference-optimization/llm-inference-metrics">Link</a></li>
</ol>
