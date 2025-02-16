# main.py
import os
import json
import pandas as pd
import gspread
from datetime import datetime, timezone

from stock_data import get_stock_data         # z.B. tägliche Kursdaten
from reddit_scraper import get_reddit_posts   # reddit posts
from sentiment_analysis import analyze_sentiment
from google_sheets import open_google_sheet, create_chart

def main():
    # 🔥 1) Google Sheet öffnen
    spreadsheet = open_google_sheet()

    # 🔥 2) Reddit Daten laden/scrapen
    reddit_data_file = "reddit_data.json"
    subreddits = ["WallStreetBets", "WallstreetbetsGer", "Mauerstrassenwetten"]
    all_posts = []

    if os.path.exists(reddit_data_file):
        with open(reddit_data_file, "r", encoding="utf-8") as f:
            all_posts = json.load(f)
        print(f"✅ {len(all_posts)} gespeicherte Reddit-Posts geladen.")
    if len(all_posts) == 0:
        for sr in subreddits:
            posts = get_reddit_posts(sr, limit=50)
            all_posts.extend(posts)
        with open(reddit_data_file, "w", encoding="utf-8") as f:
            json.dump(all_posts, f, ensure_ascii=False, indent=4)
        print(f"✅ Neue Reddit-Daten: {len(all_posts)} Posts gespeichert.")

    # 🔥 3) Aktien definieren
    stocks = ["NVDA", "AAPL", "TSLA", "MSFT"]

    for stock_name in stocks:
        print(f"\n📊 Analysiere {stock_name}...")

        # Worksheet abrufen/erstellen
        try:
            worksheet = spreadsheet.worksheet(stock_name)
            print(f"📂 Arbeitsblatt '{stock_name}' gefunden, Daten werden aktualisiert.")
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=stock_name, rows="500", cols="20")
            print(f"🆕 Neues Arbeitsblatt '{stock_name}' erstellt.")

        # 🔥 4) Sentiment pro Tag
        sentiment_data = []
        for post in all_posts:
            if "date" in post and isinstance(post["date"], (int, float)):
                # date = Unix → date()
                dt_day = datetime.fromtimestamp(post["date"], timezone.utc).date()
                val = analyze_sentiment(post["text"])  # JEDER Post
                sentiment_data.append({"Date": dt_day, "Sentiment": val})

        df_sentiment = pd.DataFrame(sentiment_data)
        if not df_sentiment.empty:
            df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"]).dt.date
            # Tagesaggregation
            df_sentiment = df_sentiment.groupby("Date").mean().reset_index()
        else:
            df_sentiment = pd.DataFrame(columns=["Date", "Sentiment"])

        # 🔥 5) Börsendaten (täglich)
        stock_data = get_stock_data(stock_name)   # Muss Spalten [Date, Close, ...] liefern

        # 🔥 6) Merge
        df_combined = pd.merge(stock_data, df_sentiment, on="Date", how="left")
        df_combined["Sentiment"] = df_combined["Sentiment"].fillna(0)

        # 🔥 7) Positive/Negative Spalten
        df_combined["Sentiment_Pos"] = df_combined["Sentiment"].apply(lambda x: x if x > 0 else 0)
        df_combined["Sentiment_Neg"] = df_combined["Sentiment"].apply(lambda x: x if x < 0 else 0)

        # 🔥 8) Achsen-Min/Max berechnen, damit Google Sheets NICHT bei 0 startet
        min_close = df_combined["Close"].min()
        max_close = df_combined["Close"].max()
        view_min = float(min_close * 0.95)  # 5% unter Minimum
        view_max = float(max_close * 1.05)  # 5% über Maximum

        # 🔥 9) In Google Sheets hochladen (4 Spalten: Date, Close, Pos, Neg)
        df_combined["Date"] = df_combined["Date"].astype(str)
        df_upload = df_combined[["Date", "Close", "Sentiment_Pos", "Sentiment_Neg"]].fillna(0)

        data_sheet = [df_upload.columns.tolist()] + df_upload.values.tolist()
        worksheet.update(data_sheet)

        # 🔥 10) Diagramm erstellen
        create_chart(worksheet, stock_name, view_min, view_max)

        print(f"✅ {stock_name} erfolgreich gespeichert!")

    print("\n🚀 Alle Analysen abgeschlossen!\n")

if __name__ == "__main__":
    main()
