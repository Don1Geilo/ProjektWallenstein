from __future__ import annotations

import json
from pathlib import Path

import duckdb

try:
    from nltk.corpus import wordnet as wn  # type: ignore
except Exception:  # pragma: no cover - fallback when nltk/wordnet missing
    wn = None  # type: ignore

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "ticker_aliases.json"


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ticker_aliases (
          ticker VARCHAR,
          alias  VARCHAR,
          source VARCHAR,
          added_at TIMESTAMP DEFAULT NOW(),
          UNIQUE(ticker, alias)
        )
    """
    )


def seed_from_json(con: duckdb.DuckDBPyConnection) -> int:
    ensure_table(con)
    if not DATA_FILE.exists():
        return 0
    data = json.loads(DATA_FILE.read_text(encoding="utf-8")) or {}
    rows = []
    for tkr, aliases in data.items():
        for a in aliases or []:
            a = (a or "").strip()
            if not a:
                continue
            rows.append((tkr.upper(), a, "seed"))
    if not rows:
        return 0
    pre_count = con.execute("SELECT COUNT(*) FROM ticker_aliases").fetchone()[0]
    con.executemany(
        "INSERT OR IGNORE INTO ticker_aliases (ticker, alias, source) VALUES (?, ?, ?)",
        rows,
    )
    post_count = con.execute("SELECT COUNT(*) FROM ticker_aliases").fetchone()[0]
    return post_count - pre_count


def add_alias(con: duckdb.DuckDBPyConnection, ticker: str, alias: str, source="manual") -> bool:
    ensure_table(con)
    t, a = ticker.upper().strip(), alias.strip()
    if not t or not a:
        return False
    con.execute(
        "INSERT OR IGNORE INTO ticker_aliases (ticker, alias, source) VALUES (?, ?, ?)",
        [t, a, source],
    )
    return True


def remove_alias(con: duckdb.DuckDBPyConnection, ticker: str, alias: str) -> int:
    ensure_table(con)
    return con.execute(
        "DELETE FROM ticker_aliases WHERE ticker = ? AND alias = ?",
        [ticker.upper().strip(), alias.strip()],
    ).rowcount


def list_aliases(con: duckdb.DuckDBPyConnection, ticker: str | None = None) -> dict[str, list[str]]:
    ensure_table(con)
    if ticker:
        rows = con.execute(
            "SELECT ticker, alias FROM ticker_aliases WHERE ticker = ? ORDER BY alias",
            [ticker.upper().strip()],
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT ticker, alias FROM ticker_aliases ORDER BY ticker, alias"
        ).fetchall()
    out: dict[str, list[str]] = {}
    for t, a in rows:
        out.setdefault(t, []).append(a)
    return out


def _synonym_variants(word: str) -> set[str]:
    """Return a set of simple synonyms for ``word`` via WordNet."""
    if wn is None:
        return set()
    try:
        syns: set[str] = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():
                    syns.add(name)
        return syns
    except Exception:  # pragma: no cover - wordnet not downloaded
        return set()


def alias_map(
    con: duckdb.DuckDBPyConnection,
    include_ticker_itself: bool = True,
    use_synonyms: bool = False,
) -> dict[str, set[str]]:
    ensure_table(con)
    rows = con.execute("SELECT ticker, alias FROM ticker_aliases").fetchall()
    m: dict[str, set[str]] = {}
    for t, a in rows:
        s = m.setdefault(t, set())
        if a:
            s.add(a)
            if use_synonyms:
                s.update(_synonym_variants(a))
        if include_ticker_itself:
            s.add(t)
            s.add(f"${t}")
    return m
