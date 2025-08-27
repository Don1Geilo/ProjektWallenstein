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
        {"id": "1", "title": "Nividia launches new GPU", "created_utc": now, "text": ""},
        {"id": "2", "title": "", "created_utc": now, "text": "I ordered something on Amzon"},
        {"id": "3", "title": "", "created_utc": now, "text": "Rheiner secures big contract"},
        {"id": "4", "title": "", "created_utc": now, "text": "Game stop hype again"},
        {"id": "5", "title": "", "created_utc": now, "text": "Ali babA expands"},
    ])

    def fake_fetch(*args, **kwargs):
        raise RuntimeError("network disabled")

    monkeypatch.setattr(reddit_scraper, "fetch_reddit_posts", fake_fetch)
    monkeypatch.setattr(reddit_scraper, "_load_posts_from_db", lambda: df)
    monkeypatch.setattr(reddit_scraper, "purge_old_posts", lambda: None)

    out = reddit_scraper.update_reddit_data(
        ["NVDA", "AMZN", "RHM", "GME", "BABA"], subreddits=None
    )
    assert len(out["NVDA"]) == 1
    assert len(out["AMZN"]) == 1
    assert len(out["RHM"]) == 1
    assert len(out["GME"]) == 1
    assert len(out["BABA"]) == 1
    assert "nividia" in out["NVDA"][0]["text"].lower()
    assert "amzon" in out["AMZN"][0]["text"].lower()
    assert "rheiner" in out["RHM"][0]["text"].lower()
    assert "game stop" in out["GME"][0]["text"].lower()
    assert "ali baba" in out["BABA"][0]["text"].lower()


def test_aliases_loaded_from_file():
    import json, importlib

    alias_path = ROOT / "data" / "ticker_aliases.json"
    original = alias_path.read_text()
    try:
        alias_path.write_text(json.dumps({"XYZ": ["some corp"]}))
        importlib.reload(reddit_scraper)
        assert reddit_scraper.TICKER_NAME_MAP == {"XYZ": ["some corp"]}
    finally:
        alias_path.write_text(original)
        importlib.reload(reddit_scraper)


def test_comment_bucketed(monkeypatch):
    now = datetime.now(timezone.utc)
    df = pd.DataFrame([
        {"id": "10", "title": "", "created_utc": now, "text": ""},
        {"id": "10_c1", "title": "", "created_utc": now, "text": "Tesler to the moon"},
    ])

    monkeypatch.setattr(reddit_scraper, "fetch_reddit_posts", lambda *a, **k: df)
    monkeypatch.setattr(reddit_scraper, "_load_posts_from_db", lambda: df)
    monkeypatch.setattr(reddit_scraper, "purge_old_posts", lambda: None)

    out = reddit_scraper.update_reddit_data(["TSLA"], subreddits=None)
    assert len(out["TSLA"]) == 1
    assert "tesler" in out["TSLA"][0]["text"].lower()


def test_fetches_hot_and_new_posts_with_comments(monkeypatch):
    now = datetime.now(timezone.utc)

    class FakeComments(list):
        def replace_more(self, limit=0):
            return None

    class FakeComment:
        def __init__(self, cid, body):
            self.id = cid
            self.body = body
            self.created_utc = now.timestamp()

    class FakePost:
        def __init__(self, pid, title, comments=None):
            self.id = pid
            self.title = title
            self.selftext = ""
            self.created_utc = now.timestamp()
            self.comments = comments or FakeComments([])

    class FakeSub:
        def hot(self, limit):
            return [FakePost("h1", "hot post", FakeComments([FakeComment("c1", "AAPL is great")]))]

        def new(self, limit):
            return [FakePost("n1", "new post")]

    class FakeReddit:
        def subreddit(self, name):
            return FakeSub()

    monkeypatch.setattr(reddit_scraper.praw, "Reddit", lambda **k: FakeReddit())

    df = reddit_scraper.fetch_reddit_posts("dummy", limit=1, include_comments=True)
    ids = set(df["id"]) if not df.empty else set()
    assert ids == {"h1", "n1", "h1_c1"}



def test_aliases_passed_as_dict(monkeypatch):
    """Custom alias mapping can be merged directly."""
    now = datetime.now(timezone.utc)
    df = pd.DataFrame([
        {"id": "1", "title": "", "created_utc": now, "text": "Some Corp news"}
    ])

    monkeypatch.setattr(reddit_scraper, "fetch_reddit_posts", lambda *a, **k: df)
    monkeypatch.setattr(reddit_scraper, "_load_posts_from_db", lambda: df)
    monkeypatch.setattr(reddit_scraper, "purge_old_posts", lambda: None)

    aliases = {"ABC": ["some corp"]}
    out = reddit_scraper.update_reddit_data(
        ["ABC"], subreddits=None, aliases=aliases
    )
    assert len(out["ABC"]) == 1
    assert "some corp" in out["ABC"][0]["text"].lower()


def test_alias_file_reloaded_each_call(monkeypatch, tmp_path):
    """Alias file is read for every update."""
    import json

    now = datetime.now(timezone.utc)
    df1 = pd.DataFrame([
        {"id": "1", "title": "", "created_utc": now, "text": "Corp1 rises"}
    ])
    df2 = pd.DataFrame([
        {"id": "2", "title": "", "created_utc": now, "text": "Corp2 rises"}
    ])

    alias_file = tmp_path / "aliases.json"
    alias_file.write_text(json.dumps({"ZZZ": ["corp1"]}))

    empty = pd.DataFrame(columns=["id", "title", "created_utc", "text"])
    monkeypatch.setattr(reddit_scraper, "fetch_reddit_posts", lambda *a, **k: empty)
    monkeypatch.setattr(reddit_scraper, "purge_old_posts", lambda: None)
    monkeypatch.setattr(reddit_scraper, "_load_posts_from_db", lambda: df1)

    out1 = reddit_scraper.update_reddit_data(
        ["ZZZ"], subreddits=None, aliases_path=alias_file
    )
    assert out1["ZZZ"] and "corp1" in out1["ZZZ"][0]["text"].lower()

    alias_file.write_text(json.dumps({"ZZZ": ["corp2"]}))
    monkeypatch.setattr(reddit_scraper, "_load_posts_from_db", lambda: df2)

    out2 = reddit_scraper.update_reddit_data(
        ["ZZZ"], subreddits=None, aliases_path=alias_file
    )
    assert out2["ZZZ"] and "corp2" in out2["ZZZ"][0]["text"].lower()


