#!/usr/bin/env python
"""CLI helper to identify trending tickers from Reddit data."""

import argparse
from wallenstein.reddit_scraper import update_reddit_data, detect_trending_tickers


def main():
    parser = argparse.ArgumentParser(description="Detect trending tickers")
    parser.add_argument("tickers", nargs="+", help="Tickers to analyse")
    parser.add_argument("--window-hours", type=int, default=24, help="Recent window in hours")
    parser.add_argument("--baseline-days", type=int, default=7, help="Baseline period in days")
    parser.add_argument("--min-mentions", type=int, default=3, help="Minimum mentions to be considered")
    parser.add_argument(
        "--ratio", type=float, default=2.0, help="Increase ratio over baseline to flag as trending"
    )
    args = parser.parse_args()

    posts = update_reddit_data(args.tickers)
    trending = detect_trending_tickers(
        posts,
        window_hours=args.window_hours,
        baseline_days=args.baseline_days,
        min_mentions=args.min_mentions,
        ratio=args.ratio,
    )
    if trending:
        print("Trending tickers:", ", ".join(trending))
    else:
        print("No trending tickers detected")


if __name__ == "__main__":
    main()
