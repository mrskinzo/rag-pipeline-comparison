import pytest
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import evaluate


def test_call_claude_score_failure(monkeypatch):
    class Dummy:
        def messages(self):
            pass

    def failing_create(*args, **kwargs):
        raise RuntimeError("api down")

    # monkeypatch the client object inside evaluate
    monkeypatch.setattr(evaluate.CLIENT, "messages", type("X", (), {"create": staticmethod(failing_create)}))
    # the helper should catch and return 0.5
    assert evaluate.call_claude_score("anything") == 0.5

    # test faithfulness with empty contexts returns a float
    res = evaluate.score_faithfulness("ans", [])
    assert isinstance(res, float)
    assert 0.0 <= res <= 1.0
