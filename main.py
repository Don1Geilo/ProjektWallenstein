import os
import json
import pandas as pd
import gspread
from datetime import datetime, timezone

# Eigene Module
from stock_data import get_stock_data            # Lädt tägliche Börsendaten
from reddit_scraper import get_reddit_posts      # Lädt Reddit-Posts
from sentiment_analysis import analyze_sentiment # Berechnet Sentiment
from google_sheets import open_google_sheet, create_chart

def main():
    # 🔥 Google Spreadsheet öffnen
    spreadsheet = open_google_sheet()

    reddit_data_file = "reddit_data.json"
    subreddits = ["WallStreetBets", "WallstreetbetsGer", "Mauerstrassenwetten"]
    all_posts = []

    # 1) Reddit-Posts laden oder neu scrapen
    if os.path.exists(reddit_data_file):
        with open(reddit_data_file, "r", encoding="utf-8") as f:
            all_posts = json.load(f)
        print(f"✅ Geladene gespeicherte Reddit-Daten ({len(all_posts)} Posts)")
    if len(all_posts) == 0:
        for sr in subreddits:
            posts = get_reddit_posts(sr, limit=50)
            all_posts.extend(posts)
        with open(reddit_data_file, "w", encoding="utf-8") as f:
            json.dump(all_posts, f, ensure_ascii=False, indent=4)
        print(f"✅ Neue Reddit-Daten gespeichert ({len(all_posts)} Posts)")

    # 🔥 Mehrere Aktien
    stocks = ["NVDA", "AAPL", "TSLA", "MSFT"]

    for stock_name in stocks:
        print(f"\n📊 Analysiere {stock_name}...")

        # 2) Worksheet vorbereiten
        try:
            worksheet = spreadsheet.worksheet(stock_name)
            print(f"📂 Arbeitsblatt '{stock_name}' gefunden. Daten werden aktualisiert.")
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=stock_name, rows="500", cols="20")
            print(f"🆕 Neues Arbeitsblatt '{stock_name}' erstellt.")

        # 3) Sentiment pro Tag ermitteln
        sentiment_data = []
        for post in all_posts:
            if "date" in post and isinstance(post["date"], (int, float)):
                # Unix → Tagesdatum
                dt_day = datetime.fromtimestamp(post["date"], timezone.utc).date()
                # JEDER Post wird analysiert (kein Ticker-Check)
                val = analyze_sentiment(post["text"])
                sentiment_data.append({"Date": dt_day, "Sentiment": val})

        df_sentiment = pd.DataFrame(sentiment_data)
        if not df_sentiment.empty:
            df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"]).dt.date
            # Tagesdurchschnitt
            df_sentiment = df_sentiment.groupby("Date").mean().reset_index()
        else:
            df_sentiment = pd.DataFrame(columns=["Date", "Sentiment"])

        # 4) Börsendaten (täglich)
        stock_data = get_stock_data(stock_name)

        # 5) Merge
        df_combined = pd.merge(stock_data, df_sentiment, on="Date", how="left")
        df_combined["Sentiment"] = df_combined["Sentiment"].fillna(0)

        # 6) Spalten für Google Sheets: 
        # A: Date | B: Close | C: Sentiment_Pos | D: Sentiment_Neg
        df_combined["Sentiment_Pos"] = df_combined["Sentiment"].apply(lambda x: x if x > 0 else 0)
        df_combined["Sentiment_Neg"] = df_combined["Sentiment"].apply(lambda x: x if x < 0 else 0)
        df_combined["Date"] = df_combined["Date"].astype(str)

        df_upload = df_combined[["Date", "Close", "Sentiment_Pos", "Sentiment_Neg"]].fillna(0)

        # 7) Daten in Google Sheets hochladen
        sheet_data = [df_upload.columns.tolist()] + df_upload.values.tolist()
        worksheet.update(sheet_data)

        # 8) Diagramm erstellen (Linie = Close, Balken = Sentiment pos/neg)
        create_chart(worksheet, stock_name)

        print(f"✅ {stock_name} erfolgreich gespeichert!")

    print("\n🚀 Alle Aktienanalysen abgeschlossen!\n")

if __name__ == "__main__":
    main()
