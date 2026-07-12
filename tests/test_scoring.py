"""Scoring behaviour.

The judge failing is not the same event as the pipeline scoring badly, and the
harness must not blur the two. call_claude_score used to swallow every
exception and return 0.5, so an API outage produced a full results file that
looked like a mediocre run. These tests pin the loud-failure behaviour.
"""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import evaluate  # noqa: E402


def _stub_judge(monkeypatch, behaviour):
    """Replace evaluate.CLIENT.messages.create with `behaviour`."""
    monkeypatch.setattr(
        evaluate.CLIENT,
        "messages",
        type("X", (), {"create": staticmethod(behaviour)}),
    )


def _reply(text):
    """Minimal stand-in for the shape of an Anthropic SDK response."""
    return type("R", (), {"content": [type("C", (), {"text": text})()]})()


def test_unreachable_judge_raises_instead_of_scoring(monkeypatch):
    def down(*args, **kwargs):
        raise RuntimeError("api down")

    _stub_judge(monkeypatch, down)
    monkeypatch.setattr(evaluate.time, "sleep", lambda *_: None)

    with pytest.raises(evaluate.ScoringError):
        evaluate.call_claude_score("anything")


def test_unparseable_score_raises(monkeypatch):
    _stub_judge(monkeypatch, lambda *a, **k: _reply("about a seven"))
    monkeypatch.setattr(evaluate.time, "sleep", lambda *_: None)

    with pytest.raises(evaluate.ScoringError):
        evaluate.call_claude_score("anything")


def test_score_followed_by_prose_is_still_a_score(monkeypatch):
    # The judge is asked for a bare decimal but sometimes justifies itself.
    # That is a real score, not a failure — the old code's float() threw on it
    # and the blanket except turned a 0.3 into a silent 0.5.
    _stub_judge(
        monkeypatch,
        lambda *a, **k: _reply("0.3\n\nThe chunk contains some information"),
    )
    assert evaluate.call_claude_score("anything") == pytest.approx(0.3)


def test_transient_failure_is_retried_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return _reply("0.8")

    _stub_judge(monkeypatch, flaky)
    monkeypatch.setattr(evaluate.time, "sleep", lambda *_: None)

    assert evaluate.call_claude_score("anything") == pytest.approx(0.8)
    assert calls["n"] == 2


def test_score_is_clamped_to_unit_interval(monkeypatch):
    _stub_judge(monkeypatch, lambda *a, **k: _reply("1.7"))
    assert evaluate.call_claude_score("anything") == 1.0


def test_faithfulness_with_no_context_is_zero_not_half():
    # Nothing retrieved means nothing in the answer can be grounded. 0.5 would
    # be an invented middling score for a pipeline that returned nothing.
    assert evaluate.score_faithfulness("an answer", []) == 0.0
