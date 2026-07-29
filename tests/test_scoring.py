import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import rag_core


class FailingClient:
    """Anthropic client stand-in whose messages.create always raises."""

    class messages:
        @staticmethod
        def create(*args, **kwargs):
            raise RuntimeError("api down")


def test_call_claude_score_failure(monkeypatch):
    # any API failure should fall back to the neutral 0.5 score
    monkeypatch.setattr(rag_core, "get_anthropic_client", lambda: FailingClient())
    assert rag_core.call_claude_score("anything") == 0.5

    # metric helpers built on it stay within [0, 1]
    res = rag_core.score_faithfulness("ans", [])
    assert isinstance(res, float)
    assert 0.0 <= res <= 1.0

    res = rag_core.score_answer_correctness("q", "ans", ["ctx"])
    assert res == 0.5


def test_call_claude_score_clamps(monkeypatch):
    class OutOfRangeClient:
        class messages:
            @staticmethod
            def create(*args, **kwargs):
                class Block:
                    text = "1.7"

                class Resp:
                    content = [Block()]

                return Resp()

    monkeypatch.setattr(rag_core, "get_anthropic_client", lambda: OutOfRangeClient())
    assert rag_core.call_claude_score("anything") == 1.0


def test_score_context_precision_empty():
    assert rag_core.score_context_precision("q", []) == 0.0
