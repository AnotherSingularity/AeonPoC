"""aeon.data.formatting — turn dataset rows into training text."""

ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)
ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{output}"
)


def format_alpaca(example: dict) -> str:
    """Format an Alpaca-style {instruction, input, output} row to a single text."""
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()
    tmpl = ALPACA_WITH_INPUT if inp else ALPACA_NO_INPUT
    return tmpl.format(instruction=instruction, input=inp, output=output)


def format_chat(messages, tokenizer=None) -> str:
    """Format a list of {role, content} messages.

    Uses the tokenizer's chat template when available; otherwise a simple
    role-prefixed fallback.
    """
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            pass
    lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
    return "\n".join(lines)
