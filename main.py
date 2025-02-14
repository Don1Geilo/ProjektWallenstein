import pandas as pd
import gspread
import json
import os
import yfinance as yf
from datetime import datetime, UTC
from oauth2client.service_account import ServiceAccountCredentials
from reddit_scraper import get_reddit_posts
from sentiment_analysis import analyze_sentiment

# 🔥 Google Sheets Verbindung einrichten
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("Google API.json", scope)
client = gspread.authorize(creds)

# 🔥 Google Spreadsheet öffnen
spreadsheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1DWOYf1KU85mJhpj_HDfIM8pqgwuS1bHKcAjSC1o_IaI/edit")

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
        posts = get_reddit_posts(subreddit, limit=100)
        all_posts.extend(posts)

    with open(reddit_data_file, "w", encoding="utf-8") as file:
        json.dump(all_posts, file, ensure_ascii=False, indent=4)
    print(f"✅ Neue Reddit-Daten gespeichert ({len(all_posts)} Posts)")

# 🔥 Mehrere Aktien auswerten
stocks = ["NVDA", "AAPL", "TSLA", "MSFT"]  # Liste der Aktien

for stock_name in stocks:
    print(f"📊 Analysiere {stock_name}...")

    # 📅 Sentiment-Daten vorbereiten
    sentiment_data = []
    for post in all_posts:
        if "date" in post and isinstance(post["date"], (int, float)):
            readable_date = datetime.fromtimestamp(post["date"], UTC).date()
            sentiment_data.append({"Date": readable_date, "Sentiment": analyze_sentiment(post["text"])})

    df_sentiment = pd.DataFrame(sentiment_data)
    df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"])
    df_sentiment = df_sentiment.groupby("Date").mean().reset_index()

    # 🔥 Tagesaktuelle Börsendaten abrufen (Yahoo Finance)
    stock_data = yf.Ticker(stock_name).history(period="7d")  # Letzte 7 Tage
    stock_data = stock_data.reset_index()
    stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.tz_localize(None)  # Zeitzone entfernen

    # 🔥 Daten zusammenführen
    df_combined = stock_data.merge(df_sentiment, on="Date", how="left")
    df_combined["Sentiment"] = df_combined["Sentiment"].fillna(0)
    df_combined["Date"] = df_combined["Date"].astype(str)

    # 🔥 Daten in Google Sheets hochladen
    worksheet = spreadsheet.add_worksheet(title=stock_name, rows="100", cols="20")
    worksheet.update([["Stock:", stock_name]] + [df_combined.columns.values.tolist()] + df_combined.values.tolist())

    print(f"✅ {stock_name} erfolgreich gespeichert!")

print("🚀 Alle Aktienanalysen abgeschlossen!")
