"""Fetch Reddit posts and store them on disk for offline analysis.

This helper script queries the Reddit API via ``wallenstein.reddit_scraper``
using the configured credentials and writes the resulting post data to a JSON
file.  The output can be used to inspect the raw data or to feed it into the
model without hitting the network again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from wallenstein import reddit_scraper


def main() -> None:
    parser = argparse.ArgumentParser(description="Store Reddit posts as JSON")
    parser.add_argument(
        "--subreddit",
        default="wallstreetbets",
        help="Subreddit to scrape",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of posts to fetch from hot and new each",
    )
    parser.add_argument(
        "--include-comments",
        dest="include_comments",
        action="store_true",
        help="Include top-level comments from hot posts (default)",
    )
    parser.add_argument(
        "--no-comments",
        dest="include_comments",
        action="store_false",
        help="Disable comment fetching",
    )
    parser.set_defaults(include_comments=True)
    parser.add_argument(
        "--comment-limit",
        type=int,
        default=3,
        help="Maximum number of comments to fetch per hot post",
    )
    parser.add_argument(
        "--output",
        default="data/reddit_posts.json",
        help="Path to output JSON file",
    )
    args = parser.parse_args()

    df = reddit_scraper.fetch_reddit_posts(
        subreddit=args.subreddit,
        limit=args.limit,
        include_comments=args.include_comments,
        comment_limit=args.comment_limit,
    )
    records = df.to_dict(orient="records")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2, default=str)

    print(f"Wrote {len(records)} posts to {out_path}")


if __name__ == "__main__":  # pragma: no cover - simple script
    main()
