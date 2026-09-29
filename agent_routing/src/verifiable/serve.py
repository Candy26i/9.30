"""Small frozen HF advisor server. Bind to loopback; no paid external API needed."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import hashlib
from pathlib import Path
import threading

from .backend import HFBackend, ContextBudgetExceeded
from .protocol import KINDS
from .sampling import normalize_generation


EXPERT_ADAPTERS = {"extractor": "default", "reasoner": "reasoner", "verifier": "verifier"}


def template_sha256():
    return hashlib.sha256(Path(__file__).with_name("chat_template.jinja").read_bytes()).hexdigest()


def load_expert_bundle(filename, base_model=None, revision=None):
    """Resolve and verify a frozen three-adapter manifest before allocating a model."""
    from .runner import checkpoint_identity
    path = Path(filename).resolve()
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("Expert bundle requires schema_version=1")
    if value.get("frozen") is not True:
        raise ValueError("Expert bundle must be frozen")
    model = value.get("base_model")
    if not isinstance(model, str) or not model:
        raise ValueError("Expert bundle requires base_model")
    if base_model is not None and model != base_model:
        raise ValueError("Expert bundle base_model differs from server configuration")
    if "base_model_revision" not in value:
        raise ValueError("Expert bundle requires base_model_revision (null only for an unpinned/local base)")
    pinned = value["base_model_revision"]
    if pinned is not None and (not isinstance(pinned, str) or not pinned):
        raise ValueError("Expert bundle base_model_revision must be a revision string or null")
    if revision is not None and pinned != revision:
        raise ValueError("Expert bundle base_model_revision differs from server configuration")
    if value.get("template_sha256") != template_sha256():
        raise ValueError("Expert bundle template fingerprint mismatch")
    if not isinstance(value.get("roles"), dict) or set(value["roles"]) != set(KINDS):
        raise ValueError("Expert bundle requires exactly extractor, reasoner and verifier roles")
    roles, paths = {}, set()
    for role in KINDS:
        entry = value["roles"][role]
        if not isinstance(entry, dict) or not isinstance(entry.get("checkpoint"), str) or not entry["checkpoint"]:
            raise ValueError(f"Expert bundle {role} requires a checkpoint path")
        checkpoint = (path.parent / entry["checkpoint"]).resolve()
        if checkpoint in paths:
            raise ValueError("Expert bundle role checkpoint paths must be distinct")
        paths.add(checkpoint)
        if not checkpoint.is_dir() or not (checkpoint / "adapter_config.json").is_file():
            raise ValueError(f"Expert bundle {role} must reference a local PEFT adapter")
        if not any((checkpoint / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            raise ValueError(f"Expert bundle {role} has no adapter weights")
        identity = checkpoint_identity(str(checkpoint))
        if not isinstance(entry.get("identity"), dict) or entry["identity"] != identity:
            raise ValueError(f"Expert bundle {role} checkpoint fingerprint mismatch")
        config = json.loads((checkpoint / "adapter_config.json").read_text())
        if config.get("peft_type") != "LORA" or config.get("base_model_name_or_path") != model:
            raise ValueError(f"Expert bundle {role} adapter base/model type mismatch")
        if config.get("revision") is not None and config["revision"] != pinned:
            raise ValueError(f"Expert bundle {role} adapter base revision mismatch")
        summary_path = checkpoint / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            if (not isinstance(summary, dict) or summary.get("training_complete") is not True
                    or summary.get("role") != role or summary.get("base_model") != model
                    or summary.get("base_model_revision") != pinned
                    or summary.get("template_sha256") != value["template_sha256"]):
                raise ValueError(f"Expert bundle {role} training summary role/completion/provenance mismatch")
        roles[role] = {"checkpoint": str(checkpoint), "identity": identity}
    return {"schema_version": 1, "base_model": model, "base_model_revision": pinned,
            "template_sha256": value["template_sha256"], "roles": roles, "frozen": True}


def expert_bundle_sha256(bundle):
    return hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FrozenExpertBackend:
    """One base model, three frozen LoRAs; switching and inference share one lock."""
    def __init__(self, bundle, max_context=16384):
        self.bundle = bundle
        self._lock = threading.Lock()
        self.backend = HFBackend(bundle["base_model"], bundle["roles"]["extractor"]["checkpoint"],
                                 max_context, revision=bundle["base_model_revision"], usage_actor="advisor")
        self.model, self.tokenizer = self.backend.model, self.backend.tokenizer
        from transformers import AutoTokenizer
        from .backend import configure_tokenizer
        from .runner import checkpoint_identity
        for role in KINDS:
            checkpoint = bundle["roles"][role]["checkpoint"]
            if role != "extractor":
                tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(checkpoint))
                if (tokenizer.get_vocab() != self.tokenizer.get_vocab()
                        or tokenizer.eos_token_id != self.tokenizer.eos_token_id):
                    raise ValueError(f"Expert bundle {role} tokenizer differs from shared base tokenizer")
                self.model.load_adapter(checkpoint, adapter_name=EXPERT_ADAPTERS[role], is_trainable=False)
            if checkpoint_identity(checkpoint) != bundle["roles"][role]["identity"]:
                raise ValueError(f"Expert bundle {role} changed while loading")
        self.model.requires_grad_(False)
        self.model.eval()

    def generate_for_role(self, role, messages, **kwargs):
        if role not in KINDS:
            raise ValueError("Unknown frozen expert role")
        with self._lock:
            adapter = EXPERT_ADAPTERS[role]
            self.model.set_adapter(adapter)
            # PEFT set_adapter may re-enable gradients on the selected LoRA.
            self.model.requires_grad_(False)
            self.model.eval()
            active = self.model.active_adapters
            if active != [adapter]:
                raise RuntimeError(f"Expert adapter selection failed: expected {adapter}, got {active}")
            result = generate_advisor_request(self.backend, {"messages": messages, **kwargs})[0]
            result["margent_role"] = {"role": role, "adapter_name": adapter,
                                      "identity": self.bundle["roles"][role]["identity"]}
            result["actual_role"] = role
            result["advisor_role"] = result["margent_role"]
            return result


def generate_advisor_request(backend, request):
    keys = {"temperature", "seed", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"}
    settings = normalize_generation({k: request[k] for k in keys if k in request})
    budget = request["max_tokens"]
    if type(budget) is not int or budget <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if isinstance(backend, FrozenExpertBackend):
        return backend.generate_for_role(request["model"], request["messages"],
                                         max_tokens=budget, **settings), settings
    try:
        result = backend.generate(request["messages"], max_tokens=budget, generation_options=settings)
    except ContextBudgetExceeded as exc:
        result = {"text": "", "prompt_tokens": exc.prompt_tokens, "completion_tokens": 0,
                  "actual_prompt_tokens": 0, "actual_completion_tokens": 0,
                  "truncated": False, "error": "context_budget_exceeded", "max_context": exc.max_context}
    return result, settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    checkpoints = p.add_mutually_exclusive_group()
    checkpoints.add_argument("--checkpoint")
    checkpoints.add_argument("--expert-bundle", help="Frozen manifest for extractor/reasoner/verifier LoRA adapters")
    p.add_argument("--revision", help="Pinned HF model revision for a frozen experiment")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-context", type=int, default=16384)
    args = p.parse_args()
    bundle = load_expert_bundle(args.expert_bundle, args.model, args.revision) if args.expert_bundle else None
    revision = bundle["base_model_revision"] if bundle is not None else args.revision
    backend = (FrozenExpertBackend(bundle, args.max_context) if bundle is not None else
               HFBackend(args.model, args.checkpoint, args.max_context, revision=revision, usage_actor="advisor"))
    from .runner import checkpoint_identity
    from .provenance import harness_identity
    fingerprint = {"model": args.model, "requested_revision": revision, "checkpoint": checkpoint_identity(args.checkpoint or args.model),
                   "harness": harness_identity(),
                   "resolved_revision": getattr(backend.model.config, "_commit_hash", None),
                   "template_sha256": template_sha256()}
    if bundle is not None:
        fingerprint.update(expert_bundle=bundle, expert_bundle_sha256=expert_bundle_sha256(bundle))

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, data):
            payload = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.send_json(200, {"status": "ready", "model": args.model,
                                 "checkpoint": args.checkpoint, "aliases": KINDS, "margent_advisor": fingerprint})

        def do_POST(self):
            try:
                if self.path != "/v1/chat/completions":
                    return self.send_json(404, {"error": "Unknown endpoint"})
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0 or length > 2_000_000:
                    return self.send_json(400, {"error": "Invalid request size"})
                request = json.loads(self.rfile.read(length))
                if request.get("model") not in KINDS:
                    return self.send_json(400, {"error": "Unknown frozen advisor alias"})
                result, settings = generate_advisor_request(backend, request)
                self.send_json(200, {"choices": [{"message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": "length" if result["truncated"] else "stop"}], "usage": {
                    "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"]},
                    "margent_advisor": fingerprint, "margent_generation": settings,
                    "margent_role": result.get("margent_role"),
                    "actual_role": (result.get("margent_role") or {}).get("role"),
                    "margent_generation_error": result.get("error"), "margent_max_context": result.get("max_context")})
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})

    print(f"Frozen advisor ready at http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
