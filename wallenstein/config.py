import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    WALLENSTEIN_DB_PATH = os.getenv("WALLENSTEIN_DB_PATH", "data/wallenstein.duckdb")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
    USE_BERT_SENTIMENT = os.getenv("USE_BERT_SENTIMENT", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
    WALLENSTEIN_TICKERS = os.getenv("WALLENSTEIN_TICKERS", "NVDA,AMZN,SMCI,TSLA")
    WALLENSTEIN_DATA_SOURCE = os.getenv("WALLENSTEIN_DATA_SOURCE", "stooq").strip().lower()
    STOOQ_USER_AGENT = os.getenv(
        "STOOQ_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    )
    YF_USER_AGENT = os.getenv(
        "YF_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    )
    DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", "30"))
    SENTIMENT_BACKEND = os.getenv("SENTIMENT_BACKEND", "finbert").lower()


settings = Settings()


def validate_config() -> None:
    required = {
        "WALLENSTEIN_DB_PATH": settings.WALLENSTEIN_DB_PATH,
        "TELEGRAM_BOT_TOKEN": settings.TELEGRAM_BOT_TOKEN,
        "TELEGRAM_CHAT_ID": settings.TELEGRAM_CHAT_ID,
        "REDDIT_CLIENT_ID": settings.REDDIT_CLIENT_ID,
        "REDDIT_CLIENT_SECRET": settings.REDDIT_CLIENT_SECRET,
        "REDDIT_USER_AGENT": settings.REDDIT_USER_AGENT,
    }

    missing = [name for name, value in required.items() if not value]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required environment variables: {missing_str}")
