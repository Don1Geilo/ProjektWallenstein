from .notify import notify_telegram


def check_alerts(df):
    """
    Beispiel-Alerts: 
    - Wenn Kurs unter 200
    - Wenn Sentiment < -2
    """
    for _, row in df.iterrows():
        if row.get("close") and row["close"] < 200:
            notify_telegram(f"⚠️ {row['ticker']} unter 200 USD gefallen! Aktuell: {row['close']}")
        if row.get("sentiment") and row["sentiment"] < -2:
            notify_telegram(f"📉 Stark negatives Sentiment bei {row['ticker']}: {row['sentiment']}")
