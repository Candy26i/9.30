import unittest

from src.manager.routing_anchor import tokenize_anchor_row


class _CharTokenizer:
    """Tiny prefix-preserving tokenizer/chat template for masking tests."""

    eos_token = "<eos>"

    @staticmethod
    def _render_message(message):
        role = message["role"]
        content = str(message.get("content") or "")
        text = f"<{role}>{content}"
        for call in message.get("tool_calls") or []:
            fn = call["function"]
            text += f"|TOOL:{fn['name']}:{fn.get('arguments', '')}"
        return text + f"</{role}>"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
        tools=None,
    ):
        del tokenize, enable_thinking, tools
        text = "".join(self._render_message(m) for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = [ord(ch) for ch in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _base_prompt():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]


class RoutingAnchorMaskTest(unittest.TestCase):
    def setUp(self):
        self.tok = _CharTokenizer()

    @staticmethod
    def _supervised_text(features):
        return "".join(chr(token) for token, label in zip(features["input_ids"], features["labels"]) if label != -100)

    def test_full_anchor_supervises_draft_and_tool_call(self):
        row = {
            "decision_type": "call",
            "prompt": _base_prompt(),
            "response": [{
                "role": "assistant",
                "content": "DRAFT_ANSWER_A",
                "tool_calls": [{
                    "id": "x",
                    "type": "function",
                    "function": {"name": "reasoner_tool", "arguments": "{}"},
                }],
            }],
        }
        features, _ = tokenize_anchor_row(row, self.tok, 4096, "full")
        supervised = self._supervised_text(features)
        self.assertIn("DRAFT_ANSWER_A", supervised)
        self.assertIn("reasoner_tool", supervised)

    def test_route_only_call_masks_draft_and_supervises_tool_call(self):
        row = {
            "decision_type": "call",
            "prompt": _base_prompt(),
            "response": [{
                "role": "assistant",
                "content": "DRAFT_ANSWER_A",
                "tool_calls": [{
                    "id": "x",
                    "type": "function",
                    "function": {"name": "reasoner_tool", "arguments": "{}"},
                }],
            }],
        }
        features, stats = tokenize_anchor_row(row, self.tok, 4096, "route_only")
        supervised = self._supervised_text(features)
        self.assertNotIn("DRAFT_ANSWER_A", supervised)
        self.assertIn("TOOL:reasoner_tool", supervised)
        self.assertGreater(stats["label_boundary"], stats["prompt_boundary"])

    def test_route_only_commit_masks_draft_and_supervises_commit_suffix(self):
        for decision_type in ("commit", "commit_after_call"):
            with self.subTest(decision_type=decision_type):
                row = {
                    "decision_type": decision_type,
                    "prompt": _base_prompt(),
                    "response": [{
                        "role": "assistant",
                        "content": "DRAFT_ANSWER_B\nANSWER_B",
                    }],
                }
                features, _ = tokenize_anchor_row(row, self.tok, 4096, "route_only")
                supervised = self._supervised_text(features)
                self.assertNotIn("DRAFT_ANSWER_B", supervised)
                self.assertIn("ANSWER_B", supervised)

    def test_route_only_calls_masks_the_draft_of_call_rows_only(self):
        call = {"decision_type": "call", "prompt": _base_prompt(),
                "response": [{"role": "assistant", "content": "DRAFT_ANSWER_A",
                              "tool_calls": [{"id": "x", "type": "function",
                                              "function": {"name": "reasoner_tool", "arguments": "{}"}}]}]}
        features, stats = tokenize_anchor_row(call, self.tok, 4096, "route_only_calls")
        supervised = self._supervised_text(features)
        self.assertNotIn("DRAFT_ANSWER_A", supervised)
        self.assertIn("TOOL:reasoner_tool", supervised)
        self.assertEqual(features["labels"], tokenize_anchor_row(call, self.tok, 4096, "route_only")[0]["labels"])
        for decision_type in ("commit", "commit_after_call"):
            with self.subTest(decision_type=decision_type):
                row = {"decision_type": decision_type, "prompt": _base_prompt(),
                       "response": [{"role": "assistant", "content": "DRAFT_ANSWER_B\nANSWER_B"}]}
                features, _ = tokenize_anchor_row(row, self.tok, 4096, "route_only_calls")
                self.assertIn("DRAFT_ANSWER_B", self._supervised_text(features))
                self.assertEqual(features["labels"], tokenize_anchor_row(row, self.tok, 4096, "full")[0]["labels"])
        with self.assertRaises(ValueError):
            tokenize_anchor_row(call, self.tok, 4096, "route_only_call")

    def test_route_only_rejects_rows_without_draft(self):
        row = {
            "decision_type": "commit",
            "prompt": _base_prompt(),
            "response": [{"role": "assistant", "content": "ANSWER_A"}],
        }
        with self.assertRaisesRegex(ValueError, "requires DRAFT_ANSWER"):
            tokenize_anchor_row(row, self.tok, 4096, "route_only")


if __name__ == "__main__":
    unittest.main()
