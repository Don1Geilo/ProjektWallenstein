"""Generate ticker alias mapping from a CSV file.

This helper reads symbol/name pairs (e.g. from a Finnhub export) and writes
``data/ticker_aliases.json`` which is used by :mod:`wallenstein.reddit_scraper`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Common corporate suffixes which are dropped for simplified aliases
STOPWORDS = {
    "inc",
    "corp",
    "corporation",
    "company",
    "co",
    "plc",
    "ag",
    "sa",
    "se",
    "nv",
    "ltd",
    "holdings",
    "holding",
    "group",
}


def _make_aliases(name: str) -> list[str]:
    """Return a list of alias variants for ``name``."""

    base = name.lower().strip()
    aliases: set[str] = {base}

    # Replace punctuation with spaces
    cleaned = re.sub(r"[^a-z0-9]+", " ", base).strip()
    if cleaned:
        aliases.add(cleaned)

    # Drop common suffixes
    tokens = [t for t in cleaned.split() if t not in STOPWORDS]
    if tokens:
        aliases.add(" ".join(tokens))
        aliases.add(tokens[0])

    return sorted(a for a in aliases if a)


def main() -> None:  # pragma: no cover - simple utility script
    parser = argparse.ArgumentParser(description="Create ticker alias mapping")
    parser.add_argument("csv", help="CSV file with at least 'symbol' and 'name' or 'description' columns")
    parser.add_argument(
        "--output",
        default=ROOT / "data" / "ticker_aliases.json",
        help="Path to output JSON file",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Determine column names
    symbol_col = next((c for c in df.columns if c.lower() in {"symbol", "ticker"}), None)
    name_col = next((c for c in df.columns if c.lower() in {"description", "name", "company"}), None)
    if not symbol_col or not name_col:
        raise SystemExit("CSV must contain symbol and name/description columns")

    mapping: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        sym = str(row[symbol_col]).upper().strip()
        name = str(row[name_col])
        mapping[sym] = _make_aliases(name)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False, indent=4, sort_keys=True)

    print(f"Wrote {len(mapping)} aliases to {out_path}")


if __name__ == "__main__":
    main()
