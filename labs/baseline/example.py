from __future__ import annotations

import sys

from rich.console import Group
from rich.live import Live
from rich.text import Text

from labs.baseline import Request, ServingSystem


def build_chat_prompt(engine: ServingSystem, user_text: str) -> str:
    return engine.tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def submit_chat_requests(engine: ServingSystem, prompts: list[str]) -> list[Request]:
    return [
        engine.submit(f"r{index}", build_chat_prompt(engine, prompt))
        for index, prompt in enumerate(prompts, start=1)
    ]


def build_renderable(engine: ServingSystem, requests: list[Request]) -> Group:
    lines: list[Text] = []
    for request in requests:
        current_text = engine.tokenizer.decode(
            request.output_ids,
            skip_special_tokens=True,
        )
        lines.append(Text(f"{request.request_id}: {current_text}", no_wrap=False))
    return Group(*lines)


def render_stream(engine: ServingSystem, requests: list[Request]) -> None:
    if sys.stdout.isatty():
        with Live(build_renderable(engine, requests), auto_refresh=False) as live:
            for _ in engine.run():
                live.update(build_renderable(engine, requests), refresh=True)
        return

    for output in engine.run():
        current_text = engine.tokenizer.decode(
            output.request.output_ids,
            skip_special_tokens=True,
        )
        print(f"{output.request.request_id}: {current_text}")


def main() -> None:
    engine = ServingSystem(
        model_name="Qwen/Qwen3-0.6B",
        max_batch_size=2,
        max_new_tokens=100,
    )

    requests = submit_chat_requests(
        engine,
        [
            "Introduce yourself in one sentence.",
            "Recommend a simple pasta idea.",
            "Describe a calm forest scene.",
            "Explain customized products briefly.",
        ],
    )
    render_stream(engine, requests)


if __name__ == "__main__":
    main()
