import os
import json
import pandas as pd
import gspread
from datetime import datetime, timezone

from stock_data import get_stock_data
from reddit_scraper import get_reddit_posts_with_comments  # Entfernte `remove_nonstock_posts`
from sentiment_analysis import analyze_sentiment
from google_sheets import open_google_sheet, create_chart
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    # 1) Google Spreadsheet öffnen
    spreadsheet = open_google_sheet()

    # 2) Reddit-Posts + Kommentare laden
    reddit_data_file = "reddit_data.json"
    subreddits = ["WallStreetBets", "WallstreetbetsGer", "Mauerstrassenwetten"]
    all_posts = []

    # Falls bereits vorhanden
    if os.path.exists(reddit_data_file):
        with open(reddit_data_file, "r", encoding="utf-8") as f:
            all_posts = json.load(f)
        print(f"✅ {len(all_posts)} gespeicherte Reddit-Posts geladen.")
    else:
        # Neu scrapen & filtern (Filterung passiert in `reddit_scraper.py`)
        all_posts = []
        for sr in subreddits:
            posts = get_reddit_posts_with_comments(sr, post_limit=500, comment_limit=25)
            all_posts.extend(posts)

        # Speichern
        with open(reddit_data_file, "w", encoding="utf-8") as f:
            json.dump(all_posts, f, ensure_ascii=False, indent=4)
        print(f"✅ Neue Reddit-Daten: {len(all_posts)} Posts (inkl. Filter).")

    # 4) Sentiment pro Tag erstellen
    sentiment_rows = []
    for post in all_posts:
        dt_day = datetime.fromtimestamp(post["date"], timezone.utc).date()
        full_text = (post["title"] or "") + " " + (post["text"] or "") + " " + " ".join(post.get("comments", []))
        val = analyze_sentiment(full_text)
        sentiment_rows.append({"Date": dt_day, "Sentiment": val})

    df_sentiment = pd.DataFrame(sentiment_rows)
    if not df_sentiment.empty:
        df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"]).dt.date
        df_sentiment = df_sentiment.groupby("Date").mean().reset_index()
    else:
        df_sentiment = pd.DataFrame(columns=["Date", "Sentiment"])

    # 5) Aktien definieren
    stocks = ["NVDA", "AAPL", "TSLA", "MSFT", "RHM.DE", "GOOGL"]

    for stock_name in stocks:
        print(f"\n📊 Analysiere {stock_name}...")

        # Worksheet
        try:
            worksheet = spreadsheet.worksheet(stock_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=stock_name, rows="500", cols="20")

        # 6) Börsendaten
        stock_data = get_stock_data(stock_name)
        stock_data["Volatility"] = stock_data["High"] - stock_data["Low"]

        # 7) Merge
        df_combined = pd.merge(stock_data, df_sentiment, on="Date", how="left")
        df_combined["Sentiment"] = df_combined["Sentiment"].fillna(0)

        # 🔥 Lags
        df_combined["Close_lag1"] = df_combined["Close"].shift(1)
        df_combined["Sentiment_lag1"] = df_combined["Sentiment"].shift(1)
        df_combined["y"] = (df_combined["Close"] > df_combined["Close_lag1"]).astype(int)
        df_combined.dropna(inplace=True)

        # 8) Train/Test
        train_size = int(len(df_combined) * 0.8)
        df_train = df_combined.iloc[:train_size].copy()
        df_test = df_combined.iloc[train_size:].copy()

        X_train = df_train[["Close_lag1", "Sentiment_lag1"]]
        y_train = df_train["y"]
        X_test = df_test[["Close_lag1", "Sentiment_lag1"]]
        y_test = df_test["y"]

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"🔍 {stock_name}: Modell-Accuracy: {acc:.2%}")

        df_test["Predicted"] = y_pred
        df_test["IsCorrect"] = (df_test["y"] == df_test["Predicted"]).astype(int)

        df_train["Predicted"] = None
        df_train["IsCorrect"] = None

        df_final = pd.concat([df_train, df_test], axis=0).sort_index()

        # 9) Positive/Negative Spalten fürs Sentiment
        df_final["Sentiment_Pos"] = df_final["Sentiment"].apply(lambda x: x if x > 0 else 0)
        df_final["Sentiment_Neg"] = df_final["Sentiment"].apply(lambda x: x if x < 0 else 0)

        # 10) Google Sheets
        df_final["Date"] = df_final["Date"].astype(str)
        df_upload = df_final[["Date", "Close", "Sentiment_Pos", "Sentiment_Neg", "Volatility", "IsCorrect"]].fillna(0)

        data_sheet = [df_upload.columns.tolist()] + df_upload.values.tolist()
        worksheet.update(data_sheet)

        # Diagram
        min_close = df_final["Close"].min()
        max_close = df_final["Close"].max()
        view_min = float(min_close * 0.95)
        view_max = float(max_close * 1.05)

        create_chart(worksheet, stock_name, view_min, view_max)
        print(f"✅ {stock_name} erfolgreich gespeichert!")

    print("\n🚀 Alle Analysen abgeschlossen!\n")

if __name__ == "__main__":
    main()
