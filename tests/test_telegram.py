"""Mock-based tests for the Telegram heartbeat helper (no network)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import telegram_notify as tn


class _FakeResp:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_noop_when_unconfigured(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert tn.telegram_configured() is False
    # must not attempt any network call
    monkeypatch.setattr(tn.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert tn.send_telegram("hi") is False


def test_sends_when_configured(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TKN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    captured = {}

    def fake_urlopen(url, data=None, timeout=None):
        captured["url"] = url
        captured["data"] = data.decode() if isinstance(data, bytes) else data
        return _FakeResp()

    monkeypatch.setattr(tn.urllib.request, "urlopen", fake_urlopen)
    assert tn.telegram_configured() is True
    assert tn.send_telegram("hello world") is True
    assert "TKN" in captured["url"]
    assert "chat_id=42" in captured["data"]
    assert "hello" in captured["data"]


def test_send_never_raises_on_network_error(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TKN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(tn.urllib.request, "urlopen", boom)
    assert tn.send_telegram("hi") is False   # swallowed, returns False


def test_format_heartbeat_contains_key_fields():
    msg = tn.format_heartbeat(
        step=14000, total_steps=30000, loss=1.847, loss_lo=1.6, loss_hi=2.4,
        mean_gate=0.044, gate_stdev=0.018, holds=True, last_audit_step=13800,
        phase_str="warm-up -> medium (next at step 7500)",
        eta_seconds=3 * 86400 + 7 * 3600, gate_delta=0.015)
    assert "14000/30000" in msg
    assert "mean|gate|: 0.044" in msg
    assert "holds=True" in msg
    assert "ETA" in msg


def test_fmt_eta():
    assert tn.fmt_eta(3 * 86400 + 7 * 3600) == "3d 7h"
    assert tn.fmt_eta(90 * 60) == "1h 30m"
    assert tn.fmt_eta(float("nan")) == "?"
