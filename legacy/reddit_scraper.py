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
STOCKS = ["NVDA", "AAPL", "TSLA", "MSFT", "RHM.DE", "GOOGL"]
REDDIT_JSON = "reddit_data.json"


def clean_old_data(file_path: str):
    """Filtert Einträge mit 'date' < NOW-RETENTION aus einer JSON-Liste heraus. Robust bei leer/korrupt."""
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        print(f"⚠️ Fehler beim Lesen von {file_path}. Datei wird zurückgesetzt.")
        return []

    if not isinstance(data, list) or not data:
        print(f"⚠️ {file_path} ist leer oder ungültig!")
        return []

    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=DATA_RETENTION_DAYS)).timestamp()

    def _safe_ts(item):
        # akzeptiert int/float/string – fällt auf 0.0 zurück
        try:
            return float(item.get("date", 0))
        except Exception:
            return 0.0

    filtered = [item for item in data if _safe_ts(item) >= cutoff_ts]

    if filtered != data:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=4, ensure_ascii=False)

    return filtered


def build_sentiment_df(posts, stocks):
    """Erzeugt pro Tag/Stock einen Sentimentwert (Mean über alle Treffer, ansonsten 0).
       Nutzt globale Synonyme zur Stock-Erkennung."""
    rows = []
    if not posts:
        return pd.DataFrame(columns=["Date", "Stock", "Sentiment"])

    for post in posts:
        try:
            dt_day = datetime.fromtimestamp(float(post.get("date", 0)), timezone.utc).date()
        except Exception:
            continue

        title = post.get("title") or ""
        text = post.get("text") or ""
        comments = post.get("comments", [])
        if not isinstance(comments, list):
            comments = []
        full_text = f"{title} {text} " + " ".join(map(str, comments))

        # Treffer je Stock auf Basis der Synonyme
        lower_full = full_text.lower()
        matched = [s for s in stocks if any(alias.lower() in lower_full
                                            for alias in global_synonyms.get(s, [s]))]
        if not matched:
            continue

        # Einmalige Sentiment-Bewertung je Post (nicht je Stock)
        val = analyze_sentiment(full_text)

        for s in matched:
            rows.append({"Date": dt_day, "Stock": s, "Sentiment": val})

    if not rows:
        return pd.DataFrame(columns=["Date", "Stock", "Sentiment"])

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    # mittlere Stimmung pro Tag/Stock
    df = df.groupby(["Date", "Stock"], as_index=False)["Sentiment"].mean()
    return df


def ensure_worksheet(spreadsheet, name: str, rows="1000", cols="50"):
    try:
        return spreadsheet.worksheet(name)
    except gspread.exceptions.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=name, rows=rows, cols=cols)


def prepare_stock_frame(stock_name: str, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Holt Kursdaten, bereitet Features auf und merged Sentiment für EINEN Stock."""
    stock_data = get_stock_data(stock_name, update_cache=True)

    if stock_data is None or stock_data.empty:
        print(f"⚠️ Keine Börsendaten für {stock_name} – überspringe.")
        return pd.DataFrame()

    # Erwartete Spalten absichern
    if "Date" not in stock_data.columns or "Close" not in stock_data.columns:
        print(f"⚠️ Stockdaten für {stock_name} ohne 'Date'/'Close'. Spalten: {list(stock_data.columns)}")
        return pd.DataFrame()

    # Date -> date (ohne Zeit), dann ggf. High/Low/Volatility
    stock_data = stock_data.copy()
    stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.date

    if "High" in stock_data.columns and "Low" in stock_data.columns:
        stock_data["Volatility"] = stock_data["High"] - stock_data["Low"]
    else:
        stock_data["Volatility"] = np.nan

    df_price = (
        stock_data.groupby("Date", as_index=False)
        .agg({"Close": "last", "High": "max", "Low": "min", "Volatility": "mean"})
    )
    # Sentiment für diesen Stock herausziehen
    df_senti = sentiment_df[sentiment_df["Stock"] == stock_name].drop(columns=["Stock"], errors="ignore").copy()
    df_senti["Date"] = pd.to_datetime(df_senti["Date"]).dt.strftime("%Y-%m-%d")
    df_price["Date"] = pd.to_datetime(df_price["Date"]).dt.strftime("%Y-%m-%d")

    # Auf denselben Zeitraum begrenzen (nur bis letztes Kursdatum)
    latest_price_day = df_price["Date"].max()
    df_senti = df_senti[df_senti["Date"] <= latest_price_day] if not df_senti.empty else df_senti

    df = pd.merge(df_price, df_senti, on="Date", how="left")
    df["Stock"] = stock_name
    # Fehlende Sentiments = 0
    df["Sentiment"] = df["Sentiment"].fillna(0.0)

    # Sortierung und Lags/Target nur für diesen Stock
    df = df.sort_values("Date")
    df["Close_lag1"] = df["Close"].shift(1)
    df["Sentiment_lag1"] = df["Sentiment"].shift(1)
    df["y"] = (df["Close"] > df["Close_lag1"]).astype(int)

    # Erst nach Lag-Bildung NAs entfernen
    df = df.dropna(subset=["Close_lag1", "Sentiment_lag1"])
    return df


def train_per_stock(df_stock: pd.DataFrame, stock_name: str):
    """Trainiert eine LogReg pro Stock (falls genug Klassen), gibt Accuracy (float/None) zurück."""
    if df_stock.empty:
        print(f"⚠️ Keine Daten für {stock_name} zum Trainieren.")
        return None

    X = df_stock[["Close_lag1", "Sentiment_lag1"]]
    y = df_stock["y"].astype(int)

    # Split 80/20 in zeitlicher Reihenfolge
    train_size = int(len(df_stock) * 0.8)
    if train_size < 1 or len(df_stock) - train_size < 1:
        print(f"⚠️ Zu wenig Datenpunkte für valides Split bei {stock_name}.")
        return None

    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

    unique_classes, counts = np.unique(y_train, return_counts=True)
    print(f"📌 y_train Klassenverteilung {stock_name}: {dict(zip(unique_classes, counts))}")

    if len(unique_classes) < 2:
        print(f"⚠️ Zu wenig Klassen in y_train für {stock_name} (nur Klasse {unique_classes[0]}), skip.")
        return None

    model = LogisticRegression(solver="liblinear", max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🔍 {stock_name}: Modell-Accuracy: {acc:.2%}")
    return float(acc)


def main():
    # --- Google Sheet vorbereiten ---
    spreadsheet = open_google_sheet()
    try:
        spreadsheet.del_worksheet(spreadsheet.worksheet("stocksOverview"))
    except gspread.exceptions.WorksheetNotFound:
        pass
    overview_ws = spreadsheet.add_worksheet(title="stocksOverview", rows="1000", cols="20")

    # --- Reddit Daten aktualisieren & laden ---
    print("🔄 Aktualisiere Reddit-Daten…")
    try:
        update_reddit_data()
    except Exception as e:
        print(f"⚠️ update_reddit_data() Fehler: {e}")

    posts = clean_old_data(REDDIT_JSON)
    print(f"✅ {len(posts)} aktuelle Reddit-Posts geladen.")

    # Sentiment-Tabelle (pro Tag/Stock)
    df_sentiment = build_sentiment_df(posts, STOCKS)

    # --- Pro Aktie verarbeiten ---
    overview_rows = []
    for stock in STOCKS:
        print(f"\n📊 Analysiere {stock}…")

        # Worksheet je Aktie
        ws = ensure_worksheet(spreadsheet, stock, rows="1000", cols="20")

        # Datenframe für EINEN Stock bauen
        df_stock = prepare_stock_frame(stock, df_sentiment)

        if df_stock.empty:
            print(f"⚠️ Keine Daten für {stock}, Überspringe Google Sheets Update & Chart.")
            continue

        # Debug-Ausgabe (letzte 10 Zeilen relevanter Spalten)
        tail_cols = ["Date", "Close", "Close_lag1", "y"]
        print(f"📌 Letzte Zeilen {stock}:")
        print(df_stock[tail_cols].tail(10))

        # Train & Accuracy
        accuracy = train_per_stock(df_stock, stock)

        # In Google Sheets schreiben
        try:
            ws.update([df_stock.columns.tolist()] + df_stock.fillna(0).values.tolist())
            print(f"✅ Daten für {stock} erfolgreich in Google Sheets aktualisiert.")
        except Exception as e:
            print(f"⚠️ Fehler beim Schreiben von {stock} in Google Sheets: {e}")

        # Overview-Zeile
        last_close = float(df_stock["Close"].iloc[-1])
        last_sent = float(df_stock["Sentiment"].iloc[-1]) if "Sentiment" in df_stock.columns else 0.0
        overview_rows.append([stock, last_close, last_sent, accuracy if accuracy is not None else ""])

        # Chart nur mit Daten dieses Stocks
        try:
            create_chart(
                worksheet=overview_ws,
                stock_name=stock,
                min_value=float(df_stock["Close"].min()) * 0.95,
                max_value=float(df_stock["Close"].max()) * 1.05,
                df=df_stock
            )
            print(f"✅ Diagramm für {stock} erfolgreich erstellt.")
        except Exception as e:
            print(f"⚠️ Fehler beim Erstellen des Diagramms für {stock}: {e}")

    # --- Overview aktualisieren ---
    if overview_rows:
        try:
            overview_ws.update([["Stock", "Last Close", "Last Sentiment", "Accuracy"]] + overview_rows)
        except Exception as e:
            print(f"⚠️ Fehler beim Schreiben der Übersicht: {e}")

    print("\n📊 Gesamtübersicht und Charts aktualisiert!")
    print("🚀 Alle Analysen abgeschlossen!\n")


if __name__ == "__main__":
    main()
