## microengine

<p align="center">
  <img src="../../assets/projects/microengine/microengine_1.png" alt="microengine diagram" width="520" />
</p>

`microengine` is a minimal inference engine built to keep the core serving components small, readable, and centered on
  the essential moving parts:

- requests
- a FIFO request queue
- a static batch scheduler
- a model runner
- one decode loop


## What it includes

- FIFO request admission
- static batching with `max_batch_size`
- one prefill pass per admitted batch
- iterative decode until each request finishes
- simple stopping rules: EOS or `max_new_tokens`
- a tiny demo in [example.py](./example.py)

## Mental model

The engine is easiest to read as this loop:

1. Submit requests into the queue.
2. The scheduler picks the next static batch.
3. Run prefill once for that batch.
4. Repeatedly decode one token step at a time.
5. Stop each request on EOS or `max_new_tokens`.
6. Move on to the next queued batch.

That's it.


## Components and responsibilities

Even across modern inference frameworks, component names and responsibilities differ a bit. There is no single source of truth for how an inference engine should be decomposed.

Because of that, `microengine` is best treated as a small mental model for understanding the serving loop holistically rather than a canonical architecture.

With that framing, `microengine` keeps the serving loop split into a few explicit pieces:

- `RequestState`
  
  Represents the lifecycle of one request: `WAITING`, `RUNNING`, or `FINISHED`.

- `Request`
  
  Holds the per-request state.
  This is the object that carries the prompt tokens, generated tokens, stopping configuration, and lifecycle state.

- `RequestQueue`
  
  Owns admission order.
  It is just a FIFO queue of waiting requests.

- `StaticBatchScheduler`
  
  Decides which requests enter the next batch.
  In this engine, the policy is intentionally simple: take the next `max_batch_size` requests from the queue and run that batch to completion.

- `BatchState`
  Holds the request states for one active batch.
  It keeps the active requests, the next input tokens, the attention mask, and the KV cache.

- `ModelRunner`
  Owns the model forward passes.
  It does two things:
  1. `prefill()` runs the prompt pass for a newly admitted batch.
  2. `decode()` advances that batch by one token.

- `MicroEngine`

  Ties the whole loop together.
  It loads the tokenizer and model, accepts requests through `submit()`, asks the scheduler for the next batch, runs prefill, and then keeps decoding until every request in that batch is done.

## Example

Run the demo directly:

```bash
python projects/microengine/example.py
```

The example constructs a `MicroEngine`, submits a few prompts, runs the engine, and prints each final decoded output.
