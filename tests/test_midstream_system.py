from __future__ import annotations

import unittest

from simple_api_router.converter_utils import fold_midstream_system_into_user
from simple_api_router.converter_openai import anthropic_to_openai_request
from simple_api_router.converter_google import anthropic_to_google_request
from simple_api_router.converter_responses import _messages_to_responses_input


# The exact shape Claude Code emits when the user types while the model is working:
# a {"role": "system"} entry right after the tool_result user turn.
QUEUED = "The user sent a new message while you were working:\n如果你看到这条消息，说2"


def _tool_loop_with_midstream_system():
    return [
        {"role": "user", "content": [{"type": "text", "text": "run it"}]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "第1秒"},
        ]},
        {"role": "system", "content": QUEUED},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]


def _roles(messages):
    return [m.get("role") for m in messages]


class TestFoldHelper(unittest.TestCase):
    def test_no_system_is_identity_noop(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
        out = fold_midstream_system_into_user(msgs)
        self.assertIs(out, msgs)  # same object — fast path, no copy

    def test_system_after_tool_result_merges_into_preceding_user(self):
        out = fold_midstream_system_into_user(_tool_loop_with_midstream_system())
        self.assertNotIn("system", _roles(out))
        # The tool_result user turn now also carries the queued text as a block.
        tr_user = out[2]
        self.assertEqual(tr_user["role"], "user")
        types = [b.get("type") for b in tr_user["content"]]
        self.assertEqual(types, ["tool_result", "text"])
        self.assertEqual(tr_user["content"][1]["text"], QUEUED)
        # No two consecutive same-role turns were introduced.
        roles = _roles(out)
        self.assertFalse(any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)))

    def test_system_after_plain_text_user_no_consecutive_user(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "note"},
            {"role": "assistant", "content": "ok"},
        ]
        out = fold_midstream_system_into_user(msgs)
        self.assertEqual(_roles(out), ["user", "assistant"])
        self.assertEqual(out[0]["content"], [{"type": "text", "text": "hello"},
                                             {"type": "text", "text": "note"}])

    def test_consecutive_system_messages_both_fold(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "u"}]},
            {"role": "system", "content": "s1"},
            {"role": "system", "content": "s2"},
            {"role": "assistant", "content": "a"},
        ]
        out = fold_midstream_system_into_user(msgs)
        self.assertEqual(_roles(out), ["user", "assistant"])
        texts = [b["text"] for b in out[0]["content"]]
        self.assertEqual(texts, ["u", "s1", "s2"])

    def test_leading_system_prepends_to_next_user(self):
        msgs = [
            {"role": "system", "content": "lead"},
            {"role": "user", "content": "hi"},
        ]
        out = fold_midstream_system_into_user(msgs)
        self.assertEqual(_roles(out), ["user"])
        self.assertEqual(out[0]["content"][0]["text"], "lead")
        self.assertEqual(out[0]["content"][1]["text"], "hi")

    def test_trailing_system_after_assistant_becomes_user(self):
        # Defensive edge (Anthropic disallows it): system after assistant, at end.
        msgs = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "system", "content": "late"},
        ]
        out = fold_midstream_system_into_user(msgs)
        self.assertEqual(_roles(out), ["user", "assistant", "user"])
        self.assertEqual(out[2]["content"][0]["text"], "late")

    def test_original_not_mutated(self):
        msgs = _tool_loop_with_midstream_system()
        before = msgs[2]["content"]
        fold_midstream_system_into_user(msgs)
        self.assertIs(msgs[2]["content"], before)  # caller's data untouched


class TestConvertersDropMidstreamSystem(unittest.TestCase):
    def test_openai_request_has_no_system_role_and_keeps_queued_text(self):
        body = {"model": "gpt-4o", "max_tokens": 100,
                "messages": _tool_loop_with_midstream_system()}
        out = anthropic_to_openai_request(body, "gpt-4o")
        roles = [m["role"] for m in out["messages"]]
        self.assertNotIn("system", roles)
        # tool_result became a tool message; queued text is a user message after it.
        self.assertIn("tool", roles)
        joined = "".join(
            m.get("content") if isinstance(m.get("content"), str) else ""
            for m in out["messages"] if m["role"] == "user"
        )
        self.assertIn("说2", joined)

    def test_google_request_queued_text_is_user_not_model(self):
        out = anthropic_to_google_request(
            {"messages": _tool_loop_with_midstream_system()}, "gemini-2.5-pro")
        contents = out["contents"]
        # The queued text must land in a user-role turn, never attributed to model.
        user_text = " ".join(
            p.get("text", "")
            for c in contents if c["role"] == "user"
            for p in c["parts"]
        )
        model_text = " ".join(
            p.get("text", "")
            for c in contents if c["role"] == "model"
            for p in c["parts"]
        )
        self.assertIn("说2", user_text)
        self.assertNotIn("说2", model_text)

    def test_responses_input_has_no_system_role(self):
        out = _messages_to_responses_input(_tool_loop_with_midstream_system())
        roles = [item.get("role") for item in out if isinstance(item, dict)]
        self.assertNotIn("system", roles)
        blob = str(out)
        self.assertIn("说2", blob)


if __name__ == "__main__":
    unittest.main()
