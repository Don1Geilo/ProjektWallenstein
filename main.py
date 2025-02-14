import os
import json
import pandas as pd
import gspread
import yfinance as yf
import config  # 🔥 Lädt API-Keys aus GitHub Secrets
from google_sheets import open_google_sheet
from reddit_scraper import get_reddit_posts
from sentiment_analysis import analyze_sentiment
from datetime import datetime, timezone

# 🚀 Debugging: Zeigt den Status der API-Keys
print(f"🚀 GOOGLE_API_KEYFILE: {config.GOOGLE_API_KEYFILE}")

# 🔥 Google Spreadsheet öffnen
spreadsheet = open_google_sheet()

# 🔥 Falls das Worksheet existiert, verwende es. Falls nicht, erstelle ein neues.
try:
    worksheet = spreadsheet.worksheet(stock_name)  # 🔍 Prüfen, ob es existiert
    print(f"📂 Arbeitsblatt '{stock_name}' gefunden. Daten werden aktualisiert.")
except gspread.exceptions.WorksheetNotFound:
    worksheet = spreadsheet.add_worksheet(title=stock_name, rows="100", cols="20")
    print(f"🆕 Neues Arbeitsblatt '{stock_name}' erstellt.")


# 🔥 Mehrere Subreddits abrufen
subreddits = ["WallStreetBets", "WallstreetbetsGer", "Mauerstrassenwetten"]
reddit_data_file = "reddit_data.json"

# 📂 Falls bereits gespeicherte Reddit-Daten existieren, laden
all_posts = []
if os.path.exists(reddit_data_file):
    with open(reddit_data_file, "r", encoding="utf-8") as file:
        all_posts = json.load(file)
    print(f"✅ Geladene gespeicherte Reddit-Daten ({len(all_posts)} Posts)")

# 🆕 Falls Datei leer ist, neue Daten scrapen
if len(all_posts) == 0:
    for subreddit in subreddits:
        posts = get_reddit_posts(subreddit, limit=100)  # 🔥 Mehr Daten sammeln
        all_posts.extend(posts)

    with open(reddit_data_file, "w", encoding="utf-8") as file:
        json.dump(all_posts, file, ensure_ascii=False, indent=4)
    print(f"✅ Neue Reddit-Daten gespeichert ({len(all_posts)} Posts)")

# 🔥 Mehrere Aktien auswerten
stocks = ["NVDA", "AAPL", "TSLA", "MSFT"]  # ✅ Aktienliste erweitern

for stock_name in stocks:
    print(f"📊 Analysiere {stock_name}...")

    # 📅 Sentiment-Daten vorbereiten
    sentiment_data = []
    for post in all_posts:
        if "date" in post and isinstance(post["date"], (int, float)):
            readable_date = datetime.fromtimestamp(post["date"], timezone.utc).date()
            sentiment_data.append({"Date": readable_date, "Sentiment": analyze_sentiment(post["text"])})

    df_sentiment = pd.DataFrame(sentiment_data)
    df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"])
    df_sentiment = df_sentiment.groupby("Date").mean().reset_index()

    # 🔥 Tagesaktuelle Börsendaten abrufen (Yahoo Finance)
    stock_data = yf.Ticker(stock_name).history(period="7d")  # 🔥 Letzte 7 Tage
    stock_data = stock_data.reset_index()
    stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.tz_localize(None)

    # 🔥 Daten zusammenführen
    df_combined = stock_data.merge(df_sentiment, on="Date", how="left")
    df_combined["Sentiment"] = df_combined["Sentiment"].fillna(0)
    df_combined["Date"] = df_combined["Date"].astype(str)

    # 🔥 Daten in Google Sheets hochladen
    worksheet = spreadsheet.add_worksheet(title=stock_name, rows="100", cols="20")
    worksheet.update([["Stock:", stock_name]] + [df_combined.columns.values.tolist()] + df_combined.values.tolist())

    print(f"✅ {stock_name} erfolgreich gespeichert!")

print("🚀 Alle Aktienanalysen abgeschlossen!")
