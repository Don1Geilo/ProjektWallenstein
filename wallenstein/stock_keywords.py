"""Synonyms for stock ticker matching."""

from __future__ import annotations

# Mapping of ticker symbols to a list of synonyms.
global_synonyms = {
    "AAPL": ["AAPL", "Apple", "MacBook", "iPhone", "Tim Cook", "iOS", "Apfel"],
    "NVDA": ["NVDA", "Nvidia", "RTX", "GeForce", "AI Chips", "GPU"],
    "TSLA": ["TSLA", "Tesla", "Elon Musk", "Cybertruck", "EV", "Model 3"],
    "MSFT": ["MSFT", "Microsoft", "Windows", "Azure", "Satya Nadella", "Xbox"],
    "RHM.DE": ["RHM.DE", "Rheiner", "Rheinmetall", "Rüstung"],
}

__all__ = ["global_synonyms"]
