import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

CACHE_FILE = "stock_data.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Fehler beim Laden des Caches – Datei wird zurückgesetzt.")
                return {}
    return {}

def save_cache(cache):
    # Stelle sicher, dass alle Timestamps in Strings konvertiert werden
    for stock in cache:
        for entry in cache[stock]:
            if "Date" in entry:
                entry["Date"] = str(entry["Date"])
    
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)

def get_stock_data(stock_name, update_cache=True):
    cache = load_cache()
    now = datetime.now(timezone.utc)
    seven_days_ago = now - pd.Timedelta(days=7)

    # Falls Cache vorhanden ist, verwende ihn
    if stock_name in cache and not update_cache:
        stock_data = pd.DataFrame(cache[stock_name])
        stock_data["Date"] = pd.to_datetime(stock_data["Date"])
        stock_data["Date"] = stock_data["Date"].dt.tz_localize(None)  # Entferne Zeitzone
        return stock_data

    print(f"📥 Lade Börsendaten für {stock_name}...")
    stock = yf.Ticker(stock_name)
    stock_data = stock.history(period="7d", interval="1h")

    if stock_data.empty:
        print(f"⚠️ Keine Daten für {stock_name}")
        return pd.DataFrame()

    # Reset Index, um sicherzustellen, dass das Datum als Spalte existiert
    stock_data.reset_index(inplace=True)
    stock_data["Date"] = stock_data["Datetime"]
    stock_data.drop(columns=["Datetime"], inplace=True, errors="ignore")
    
    # Konvertiere `Date` in UTC und String-Format für JSON-Speicherung
    stock_data["Date"] = stock_data["Date"].dt.tz_convert("UTC")
    stock_data["Date"] = stock_data["Date"].dt.tz_localize(None)  # Entferne Zeitzone
    stock_data["Date"] = stock_data["Date"].astype(str)

    # Update Cache
    cache[stock_name] = stock_data.to_dict(orient="records")
    save_cache(cache)
    
    return stock_data
