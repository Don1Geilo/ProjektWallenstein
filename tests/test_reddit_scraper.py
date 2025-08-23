import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Konfiguration stubben, damit beim Import keine echten Secrets nötig sind
os.environ.setdefault("CLIENT_ID", "test")
os.environ.setdefault("CLIENT_SECRET", "test")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")

from wallenstein import reddit_scraper


def test_company_name_bucketed(monkeypatch):
    now = datetime.now(timezone.utc)
    df = pd.DataFrame([
        {"id": "1", "title": "Nvidia launches new GPU", "created_utc": now, "text": ""},
        {"id": "2", "title": "", "created_utc": now, "text": "I ordered something on Amazon"},
    ])

    def fake_fetch(*args, **kwargs):
        raise RuntimeError("network disabled")

    monkeypatch.setattr(reddit_scraper, "fetch_reddit_posts", fake_fetch)
    monkeypatch.setattr(reddit_scraper, "_load_posts_from_db", lambda: df)
    monkeypatch.setattr(reddit_scraper, "purge_old_posts", lambda: None)

    out = reddit_scraper.update_reddit_data(["NVDA", "AMZN"], subreddits=None)
    assert len(out["NVDA"]) == 1
    assert len(out["AMZN"]) == 1
    assert "nvidia" in out["NVDA"][0]["text"].lower()
    assert "amazon" in out["AMZN"][0]["text"].lower()
