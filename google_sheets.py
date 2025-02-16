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


def delete_existing_charts(sheet):
    """Löscht alle vorhandenen Diagramme in Google Sheets"""
    requests = []
    spreadsheet_metadata = sheet.spreadsheet.fetch_sheet_metadata()
    
    for sheet_data in spreadsheet_metadata["sheets"]:
        if sheet_data["properties"]["title"] == sheet.title and "charts" in sheet_data:
            for chart in sheet_data["charts"]:
                requests.append({"deleteEmbeddedObject": {"objectId": chart["chartId"]}})

    if requests:
        sheet.spreadsheet.batch_update({"requests": requests})
        print("🗑️ Alte Diagramme wurden gelöscht!")

def create_chart(sheet, stock_name):
    """Erstellt ein Kurs+Sentiment-Diagramm (Linie + Balken) in Google Sheets."""

    # 1. Bestehende Charts entfernen
    delete_existing_charts(sheet)

    sheet_id = sheet.spreadsheet.worksheet(sheet.title)._properties['sheetId']

    # 2. Definiere das Kombi-Diagramm
    body = {
        "requests": [
            {
                "addChart": {
                    "chart": {
                        "spec": {
                            "title": f"{stock_name} -Kurs & Reddit Sentiment",
                            "basicChart": {
                                "chartType": "COMBO",
                                "legendPosition": "BOTTOM_LEGEND",
                                # Achsen: links = Kurs, rechts = Sentiment
                                "axis": [
                                    {"position": "BOTTOM_AXIS", "title": "Datum"},
                                    {"position": "LEFT_AXIS", "title": f"{stock_name} Kurs"},
                                    {"position": "RIGHT_AXIS", "title": "Sentiment"},
                                ],
                                # X-Achse = Spalte A (Index 0)
                                "domains": [
                                    {
                                        "domain": {
                                            "sourceRange": {
                                                "sources": [
                                                    {
                                                        "sheetId": sheet_id,
                                                        "startRowIndex": 1,
                                                        "endRowIndex": 500,
                                                        "startColumnIndex": 0,
                                                        "endColumnIndex": 1
                                                    }
                                                ]
                                            }
                                        }
                                    }
                                ],
                                "series": [
                                    {  # 🔹 Kurs (blaue Linie) = Spalte B (Index=1)
                                        "series": {
                                            "sourceRange": {
                                                "sources": [
                                                    {
                                                        "sheetId": sheet_id,
                                                        "startRowIndex": 1,
                                                        "endRowIndex": 500,
                                                        "startColumnIndex": 1,
                                                        "endColumnIndex": 2
                                                    }
                                                ]
                                            }
                                        },
                                        "targetAxis": "LEFT_AXIS",
                                        "type": "LINE",
                                        "color": {"red": 0.0, "green": 0.0, "blue": 1.0}  # Blau
                                    },
                                    {  # 🔹 Sentiment Positive (grüne Balken) = Spalte C (Index=2)
                                        "series": {
                                            "sourceRange": {
                                                "sources": [
                                                    {
                                                        "sheetId": sheet_id,
                                                        "startRowIndex": 1,
                                                        "endRowIndex": 500,
                                                        "startColumnIndex": 2,
                                                        "endColumnIndex": 3
                                                    }
                                                ]
                                            }
                                        },
                                        "targetAxis": "RIGHT_AXIS",
                                        "type": "COLUMN",
                                        "color": {"red": 0.0, "green": 1.0, "blue": 0.0}  # Grün
                                    },
                                    {  # 🔹 Sentiment Negative (rote Balken) = Spalte D (Index=3)
                                        "series": {
                                            "sourceRange": {
                                                "sources": [
                                                    {
                                                        "sheetId": sheet_id,
                                                        "startRowIndex": 1,
                                                        "endRowIndex": 500,
                                                        "startColumnIndex": 3,
                                                        "endColumnIndex": 4
                                                    }
                                                ]
                                            }
                                        },
                                        "targetAxis": "RIGHT_AXIS",
                                        "type": "COLUMN",
                                        "color": {"red": 1.0, "green": 0.0, "blue": 0.0}  # Rot
                                    },
                                ]
                            }
                        },
                        # Diagramm-Position
                        "position": {
                            "overlayPosition": {
                                "anchorCell": {
                                    "sheetId": sheet_id,
                                    "rowIndex": 1,
                                    "columnIndex": 8
                                }
                            }
                        }
                    }
                }
            }
        ]
    }

    # 3. An Google Sheets senden
    sheet.spreadsheet.batch_update(body)
    print(f"📈 Diagramm für {stock_name} erstellt!")






print("Diagramm erstellt!")
   


