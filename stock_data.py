import yfinance as yf
import pandas as pd

def get_stock_data(stock_name):
    """
    Holt tägliche Börsendaten von Yahoo Finance (14 Tage).
    Gibt ein DataFrame mit 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'.
    """
    stock = yf.Ticker(stock_name)
    # 🔥 Hier nur Tagesbasis statt stündlich
    stock_data = stock.history(period="14d", interval="1d")
    
    if stock_data.empty:
        raise ValueError(f"❌ Keine Daten für {stock_name} erhalten!")

    # Index → Spalte
    stock_data = stock_data.reset_index()
    # Mögliche Umbenennung "Date" vs. "Datetime"
    if "Datetime" in stock_data.columns:
        stock_data.rename(columns={"Datetime": "Date"}, inplace=True)

    if "Date" not in stock_data.columns:
        raise KeyError(f"❌ 'Date'-Spalte fehlt nach Reset bei {stock_name}!")

    # Nur Datum, keine Uhrzeit
    stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.date

    # Nur relevante Spalten
    return stock_data[["Date", "Open", "High", "Low", "Close", "Volume"]]
