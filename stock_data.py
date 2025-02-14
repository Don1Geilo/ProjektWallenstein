import yfinance as yf

def get_stock_data(ticker, period="1mo"):
    """Holt historische Börsendaten für eine Aktie."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df[["Close"]]
