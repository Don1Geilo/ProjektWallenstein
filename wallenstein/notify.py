import logging
import os
from typing import Iterable, List

import requests

from wallenstein.config import settings

log = logging.getLogger(__name__)

TELEGRAM_MAX_LENGTH = 3900  # reserve a small safety margin below 4096 chars


def _split_message(text: str, limit: int = TELEGRAM_MAX_LENGTH) -> List[str]:
    """Split ``text`` into Telegram-safe chunks of ``limit`` characters."""

    if len(text) <= limit:
        return [text]

    lines = text.splitlines()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def _flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n".join(current).strip())
            current = []
            current_len = 0

    for line in lines:
        if not line:
            extra = 1  # newline
        else:
            extra = len(line) + 1
        if current and current_len + extra > limit:
            _flush()
        if len(line) > limit:
            for start in range(0, len(line), limit):
                segment = line[start : start + limit]
                if segment:
                    chunks.append(segment)
            current = []
            current_len = 0
            continue
        current.append(line)
        current_len += extra

    _flush()
    return [chunk for chunk in chunks if chunk]


def _send_chunks(token: str, chat_id: str, chunks: Iterable[str]) -> bool:
    all_ok = True
    for idx, chunk in enumerate(chunks, start=1):
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"},
            )
            ok = r.ok and r.json().get("ok", False)
            if ok:
                log.info("✅ Telegram gesendet (Teil %s).", idx)
            else:
                log.warning("⚠️ Telegram-API Response (Teil %s): %s", idx, r.text)
            all_ok = all_ok and ok
        except Exception as e:  # pragma: no cover - network error
            log.error("❌ Telegram-Error (Teil %s): %s", idx, e)
            print(f"❌ Telegram-Error: {e}")
            all_ok = False
    return all_ok


def notify_telegram(text: str) -> bool:

    token = settings.TELEGRAM_BOT_TOKEN or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = settings.TELEGRAM_CHAT_ID or os.getenv("TELEGRAM_CHAT_ID")

    if not text or not token or not chat_id:
        log.warning("⚠️ Telegram nicht konfiguriert oder Chat-ID fehlt.")
        return False

    chunks = _split_message(text)
    if len(chunks) > 1:
        log.info("Nachricht wird in %s Teile gesplittet (Länge %s).", len(chunks), len(text))

    return _send_chunks(token, chat_id, chunks)
