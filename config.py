import os
from dotenv import load_dotenv

# 🔥 .env Datei laden
load_dotenv()

# 🔥 Reddit API Credentials
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")

# 🔥 Google API
GOOGLE_API_KEYFILE = os.getenv("GOOGLE_API_KEYFILE")
GOOGLE_SHEETS_ID = os.getenv("GOOGLE_SHEETS_ID")  # ✅ Google Sheets ID aus .env Datei laden

# 🔥 Debug
print(f"🚀 GOOGLE_API_KEYFILE: {GOOGLE_API_KEYFILE}")  # Falls None → .env nicht richtig geladen!

