import gspread
from oauth2client.service_account import ServiceAccountCredentials
import config
from pathlib import Path

# 🔥 Absoluten Pfad zur JSON-Datei bestimmen
google_api_path = Path(config.GOOGLE_API_KEYFILE).resolve()

# 🔥 Google Sheets Verbindung einrichten
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(str(google_api_path), scope)
client = gspread.authorize(creds)

print("✅ Google Sheets Verbindung erfolgreich!")

# 🔥 Funktion zum Öffnen des Google Sheets
def open_google_sheet(sheet_id=None):
    """Öffnet das Google Spreadsheet anhand einer Google Sheet ID."""
    if sheet_id is None:
        sheet_id = config.GOOGLE_SHEETS_ID  # Falls keine ID übergeben wird, Standardwert nehmen
    spreadsheet = client.open_by_key(sheet_id)
    return spreadsheet  # ❗ Jetzt geben wir das gesamte Spreadsheet zurück


