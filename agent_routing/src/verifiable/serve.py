"""Small frozen HF advisor server. Bind to loopback; no paid external API needed."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import hashlib
from pathlib import Path

from .backend import HFBackend, ContextBudgetExceeded
from .protocol import KINDS
from .sampling import normalize_generation


def generate_advisor_request(backend, request):
    keys = {"temperature", "seed", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"}
    settings = normalize_generation({k: request[k] for k in keys if k in request})
    budget = request["max_tokens"]
    if type(budget) is not int or budget <= 0:
        raise ValueError("max_tokens must be a positive integer")
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
    p.add_argument("--checkpoint")
    p.add_argument("--revision", help="Pinned HF model revision for a frozen experiment")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-context", type=int, default=16384)
    args = p.parse_args()
    backend = HFBackend(args.model, args.checkpoint, args.max_context, revision=args.revision)
    from .runner import checkpoint_identity
    from .provenance import harness_identity
    fingerprint = {"model": args.model, "requested_revision": args.revision, "checkpoint": checkpoint_identity(args.checkpoint or args.model),
                   "harness": harness_identity(),
                   "resolved_revision": getattr(backend.model.config, "_commit_hash", None),
                   "template_sha256": hashlib.sha256(Path(__file__).with_name("chat_template.jinja").read_bytes()).hexdigest()}

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
                    "margent_generation_error": result.get("error"), "margent_max_context": result.get("max_context")})
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})

    print(f"Frozen advisor ready at http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
