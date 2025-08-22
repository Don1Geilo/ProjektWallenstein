import os
import json
import pandas as pd
import gspread
from datetime import datetime, timezone, timedelta
import numpy as np
from stock_data import get_stock_data
from reddit_scraper import update_reddit_data
from sentiment_analysis import analyze_sentiment
from google_sheets import open_google_sheet, create_chart
from stock_keywords import global_synonyms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_RETENTION_DAYS = 7  

def clean_old_data(file_path):
    """Löscht alte Daten aus der JSON-Datei, die älter als 7 Tage sind."""
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        print(f"⚠️ Fehler beim Lesen von {file_path}. Datei wird zurückgesetzt.")
        data = []

    if not data:
        print(f"⚠️ {file_path} ist leer oder ungültig!")
        return []

    cutoff_timestamp = (datetime.now(timezone.utc) - timedelta(days=DATA_RETENTION_DAYS)).timestamp()
    filtered_data = [item for item in data if float(item.get("date", 0)) >= cutoff_timestamp]

    if filtered_data != data:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, indent=4, default=lambda o: int(o) if isinstance(o, np.integer) else o)

    return filtered_data

def main():
    spreadsheet = open_google_sheet()

    try:
        spreadsheet.del_worksheet(spreadsheet.worksheet("stocksOverview"))
    except gspread.exceptions.WorksheetNotFound:
        pass
    overview_worksheet = spreadsheet.add_worksheet(title="stocksOverview", rows="500", cols="20")

    print("🔄 Aktualisiere Reddit-Daten...")
    update_reddit_data()
    reddit_data_file = "reddit_data.json"
    all_posts = clean_old_data(reddit_data_file)

    print(f"✅ {len(all_posts)} aktuelle Reddit-Posts geladen.")

    stocks = ["NVDA", "AAPL", "TSLA", "MSFT", "RHM.DE", "GOOGL"]

    sentiment_rows = []
    
    for post in all_posts:
        dt_day = datetime.fromtimestamp(float(post["date"]), timezone.utc).date()
        full_text = (post["title"] or "") + " " + (post["text"] or "") + " ".join(post.get("comments", []))

        matched_stocks = []
        for stock_name in stocks:
            if any(alias.lower() in full_text.lower() for alias in global_synonyms.get(stock_name, [stock_name])):  
                matched_stocks.append(stock_name)

        if matched_stocks:
            val = analyze_sentiment(full_text)
            for stock_name in matched_stocks:
                sentiment_rows.append({"Date": dt_day, "Stock": stock_name, "Sentiment": val})

    df_sentiment = pd.DataFrame(sentiment_rows)

    if not df_sentiment.empty:
        df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"])
        df_sentiment = df_sentiment.groupby(["Date", "Stock"]).mean().reset_index()
    else:
        df_sentiment = pd.DataFrame(columns=["Date", "Stock", "Sentiment"])

    overview_data = []
    stock_charts = []

    for stock_name in stocks:
        print(f"\n📊 Analysiere {stock_name}...")

        try:
            worksheet = spreadsheet.worksheet(stock_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=stock_name, rows="500", cols="20")

        stock_data = get_stock_data(stock_name, update_cache=True)

        if stock_data.empty:
            print(f"⚠️ Keine Börsendaten für {stock_name} – überspringe.")
            continue

        stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.date

        if "High" in stock_data.columns and "Low" in stock_data.columns:
            stock_data["Volatility"] = stock_data["High"] - stock_data["Low"]
        else:
            print(f"⚠️ Spalten 'High' und 'Low' fehlen – Volatility kann nicht berechnet werden.")
            stock_data["Volatility"] = np.nan

        stock_data = stock_data.groupby("Date").agg({
            "Close": "last", "High": "max", "Low": "min", "Volatility": "mean"
        }).reset_index()

        df_sentiment_stock = df_sentiment[df_sentiment["Stock"] == stock_name].drop(columns=["Stock"], errors="ignore")
        df_sentiment_stock["Date"] = pd.to_datetime(df_sentiment_stock["Date"]).dt.strftime("%Y-%m-%d")
        stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.strftime("%Y-%m-%d")

        latest_stock_date = stock_data["Date"].max()
        df_sentiment_stock = df_sentiment_stock[df_sentiment_stock["Date"] <= latest_stock_date]
        df_temp = pd.merge(stock_data, df_sentiment_stock, on="Date", how="left")
        df_temp["Stock"] = stock_name  # 🔥 Stock-Spalte wieder hinzufügen

        if 'df_combined' in locals():
            df_combined = pd.concat([df_combined, df_temp], ignore_index=True)  # 🔥 Daten für alle Aktien speichern
        else:
            df_combined = df_temp.copy()  # Erste Initialisierung


        df_combined["Sentiment"] = df_combined["Sentiment"].fillna(0)

        df_combined["Close_lag1"] = df_combined["Close"].shift(1)
        df_combined["Sentiment_lag1"] = df_combined["Sentiment"].shift(1)
        df_combined["y"] = (df_combined["Close"] > df_combined["Close_lag1"]).astype(int)
        df_combined.dropna(inplace=True)

        print(f"📌 Letzte 10 Zeilen von Close, Close_lag1 und y für {stock_name}:")
        print(df_combined[["Close", "Close_lag1", "y"]].tail(10))

        train_size = int(len(df_combined) * 0.8)
        df_train = df_combined.iloc[:train_size]
        df_test = df_combined.iloc[train_size:]

        X_train = df_train[["Close_lag1", "Sentiment_lag1"]]
        y_train = df_train["y"]
        X_test = df_test[["Close_lag1", "Sentiment_lag1"]]
        y_test = df_test["y"]

        unique_classes, counts = np.unique(y_train, return_counts=True)
        print(f"📌 y_train Klassenverteilung für {stock_name}: {dict(zip(unique_classes, counts))}")

        if len(unique_classes) < 2:
            print(f"⚠️ Zu wenig Klassen in y_train für {stock_name} (nur {unique_classes[0]}), überspringe Modell-Training.")
            accuracy = None
        else:
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"🔍 {stock_name}: Modell-Accuracy: {accuracy:.2%}")

        df_stock = df_combined[df_combined["Stock"] == stock_name]  # 🔥 Filtere nur die Daten der aktuellen Aktie

        if df_stock.empty:
            print(f"⚠️ Keine Daten für {stock_name}, Überspringe Google Sheets Update.")
        else:
            worksheet.update([df_stock.columns.tolist()] + df_stock.fillna(0).values.tolist())
            print(f"✅ Daten für {stock_name} erfolgreich in Google Sheets aktualisiert.")


        stock_charts.append((overview_worksheet, stock_name, df_combined["Close"].min() * 0.95, df_combined["Close"].max() * 1.05))
        overview_data.append([stock_name, df_combined["Close"].iloc[-1], df_combined["Sentiment"].iloc[-1], accuracy])

    overview_worksheet.update([["Stock", "Last Close", "Last Sentiment", "Accuracy"]] + overview_data)
    for stock_name in stocks:
        print(f"📊 Erstelle Diagramm für {stock_name} in stocksOverview...")

        # 🔥 Korrekte Daten nur für die aktuelle Aktie filtern
        df_stock = df_combined[df_combined["Stock"] == stock_name].copy()

        if df_stock.empty:
            print(f"⚠️ Keine Daten für {stock_name}, Diagramm übersprungen.")
            continue

        print(f"📌 Daten für {stock_name} zur Diagrammerstellung:")
        print(df_stock[["Date", "Stock", "Close"]].tail())  # Debugging: Zeigt letzte Werte

        try:
            # 🔥 Stelle sicher, dass create_chart NUR die Daten der jeweiligen Aktie nutzt
            create_chart(
                worksheet=overview_worksheet, 
                stock_name=stock_name, 
                min_value=df_stock["Close"].min() * 0.95, 
                max_value=df_stock["Close"].max() * 1.05,
                df=df_stock  # NUR die aktuelle Aktie übergeben
            )
            print(f"✅ Diagramm für {stock_name} erfolgreich erstellt.")
        except Exception as e:
            print(f"⚠️ Fehler beim Erstellen des Diagramms für {stock_name}: {e}")

    print("📊 Gesamtübersicht und Charts aktualisiert!")
    print("\n🚀 Alle Analysen abgeschlossen!\n")

if __name__ == "__main__":
    main()
