"""
scripts/telegram_notify.py — optional Telegram heartbeat/alerts for Stage 2.

Token and chat id are read from the environment (TELEGRAM_BOT_TOKEN,
TELEGRAM_CHAT_ID). If either is unset, sending is a no-op that returns False,
so training never depends on telemetry being configured. Uses only the stdlib
(urllib) so it has no extra dependency and is trivially mockable in tests.
"""
import os
import urllib.request
import urllib.parse

API = "https://api.telegram.org/bot{token}/sendMessage"


def telegram_configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN")
               and os.environ.get("TELEGRAM_CHAT_ID"))


def send_telegram(text: str, timeout: float = 10.0) -> bool:
    """POST `text` to the configured chat. Returns True on HTTP 200.

    Never raises — telemetry must not be able to crash a multi-week run.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return False
    url = API.format(token=token)
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=timeout) as resp:
            status = getattr(resp, "status", None)
            if status is None:
                status = resp.getcode()
            return status == 200
    except Exception as e:  # noqa: BLE001 - telemetry must never crash training
        print(f"[telegram] send failed: {e}")
        return False


def fmt_eta(seconds: float) -> str:
    if seconds is None or seconds != seconds or seconds < 0:  # None or NaN
        return "?"
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


def format_heartbeat(step, total_steps, loss, loss_lo, loss_hi, mean_gate,
                     gate_stdev, holds, last_audit_step, phase_str,
                     eta_seconds, gate_delta=None) -> str:
    pct = 100.0 * step / total_steps if total_steps else 0.0
    delta = f" (Δ {gate_delta:+.3f} since Stage 1 start)" if gate_delta is not None else ""
    return (
        f"Aeon Stage 2 — step {step}/{total_steps} "
        f"({pct:.1f}%, ETA {fmt_eta(eta_seconds)})\n"
        f"loss: {loss:.3f} (range {loss_lo:.1f}-{loss_hi:.1f} last window)\n"
        f"mean|gate|: {mean_gate:.3f}{delta}\n"
        f"gate stdev across layers: {gate_stdev:.3f}\n"
        f"certs: holds={holds} (last audit step {last_audit_step})\n"
        f"phase: {phase_str}"
    )


def format_alert(step, message) -> str:
    return f"⚠️ Aeon Stage 2 ALERT @ step {step}\n{message}"
