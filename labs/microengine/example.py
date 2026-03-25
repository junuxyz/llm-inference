from __future__ import annotations

from microengine import ServingSystem


def main() -> None:
    engine = ServingSystem(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_batch_size=2,
        max_new_tokens=40,
    )

    requests = [
        engine.submit("r1", "I am"),
        engine.submit("r2", "I like"),
        engine.submit("r3", "I love"),
        engine.submit("r4", "I do"),
    ]
    engine.run()

    for request in requests:
        print(
            f"{request.request_id}: "
            f"{engine.tokenizer.decode([*request.prompt_ids, *request.output_ids], skip_special_tokens=True)}"
        )


if __name__ == "__main__":
    main()
