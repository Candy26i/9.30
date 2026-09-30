"""Opt-in finite decision grammar, with identical generation/scoring support.

Restricts syntax and repeated tool use only. It never reads answers/rewards or
changes the stored solution. Revisions remain unconstrained language generation.
Legal actions are the exact assistant text the fixed chat template renders, so
the grammar admits the same tokens that Manager SFT supervises.
"""
from .protocol import COMMIT, KINDS, call_message


class ActionTrie:
    def __init__(self, paths):
        self.paths = [list(path) for path in paths]
        if not self.paths or any(not p for p in self.paths):
            raise ValueError("Empty action grammar")
        self.next = {}
        for path in self.paths:
            for index, token in enumerate(path):
                self.next.setdefault(tuple(path[:index]), set()).add(token)

    def allowed(self, prefix):
        choices = self.next.get(tuple(prefix))
        if not choices:
            raise ValueError("Generated prefix is outside the finite action grammar")
        return sorted(choices)

    def generation_kwargs(self, prompt_length):
        return {"prefix_allowed_tokens_fn": lambda batch, ids: self.allowed(ids[prompt_length:].tolist())}


def rendered_action(tokenizer, messages, tools, response):
    """Assistant text rendered for one decision turn, without its ChatML ending.

    SFT targets come from the same rendering (training.tokenize_turn). A tool call
    renders as a leading newline plus default-separator JSON; a hand-built compact
    string would make that supervised first token unreachable at decode time.
    """
    from .backend import render
    prompt = render(tokenizer, messages, tools)
    full = render(tokenizer, messages + [response], tools, generation=False)
    ending = tokenizer.eos_token + "\n"
    if not full.startswith(prompt) or not full.endswith(ending) or len(full) == len(prompt) + len(ending):
        raise ValueError("Chat template does not render a prefix-preserving decision turn")
    return full[len(prompt):-len(ending)]


def decision_paths(tokenizer, messages, tools, budget):
    names = {tool["function"]["name"] for tool in tools}
    if not names <= {kind + "_tool" for kind in KINDS}:
        raise ValueError("Unknown tools in finite action grammar")
    used = set()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls", []):
            used.add(call["function"]["name"])
    responses = [{"role": "assistant", "content": COMMIT}] + [
        call_message(name.removesuffix("_tool"), None, "call") for name in sorted(names - used)]
    actions = [rendered_action(tokenizer, messages, tools, response) for response in responses]
    paths = []
    for action in actions:
        ids = tokenizer(action, add_special_tokens=False)["input_ids"]
        if tokenizer.decode(ids, skip_special_tokens=False) != action:
            raise ValueError("Tokenizer cannot round-trip the canonical action")
        if tokenizer.eos_token_id in ids:
            raise ValueError("EOS inside action")
        paths.append(ids + [tokenizer.eos_token_id])
    if max(map(len, paths)) > budget:
        raise ValueError("Decision budget too small for a complete legal action plus EOS")
    return paths
