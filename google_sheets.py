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
    """Erstellt ein neues Diagramm in Google Sheets"""
    
    # 🔥 Zuerst alle alten Diagramme entfernen
    delete_existing_charts(sheet)

    body = {
        "requests": [
            {
                "addChart": {
                    "chart": {
                        "spec": {
                            "title": f"{stock_name} Kurs & Sentiment",
                            "basicChart": {
                                "chartType": "LINE",
                                "legendPosition": "BOTTOM_LEGEND",
                                "axis": [
                                    {"position": "BOTTOM_AXIS", "title": "Datum"},
                                    {"position": "LEFT_AXIS", "title": "Börsenkurs"},
                                    {"position": "RIGHT_AXIS", "title": "Sentiment"},
                                ],
                                "domains": [
                                    {
                                        "domain": {
                                            "sourceRange": {
                                                "sources": [
                                                    {"sheetId": sheet.spreadsheet.worksheet(sheet.title)._properties['sheetId'], 
                                                     "startRowIndex": 1, 
                                                     "endRowIndex": 100, 
                                                     "startColumnIndex": 0, 
                                                     "endColumnIndex": 1}
                                                ]
                                            }
                                        }
                                    }
                                ],
                                "series": [
                                    {
                                        "series": {
                                            "sourceRange": {
                                                "sources": [
                                                    {"sheetId": sheet.spreadsheet.worksheet(sheet.title)._properties['sheetId'], 
                                                     "startRowIndex": 1, 
                                                     "endRowIndex": 100, 
                                                     "startColumnIndex": 1, 
                                                     "endColumnIndex": 2}
                                                ]
                                            }
                                        },
                                        "targetAxis": "LEFT_AXIS",
                                    },
                                    {
                                        "series": {
                                            "sourceRange": {
                                                "sources": [
                                                    {"sheetId": sheet.spreadsheet.worksheet(sheet.title)._properties['sheetId'], 
                                                     "startRowIndex": 1, 
                                                     "endRowIndex": 100, 
                                                     "startColumnIndex": 2, 
                                                     "endColumnIndex": 3}
                                                ]
                                            }
                                        },
                                        "targetAxis": "RIGHT_AXIS",
                                    }
                                ],
                            }
                        },
                       "position": {
    "overlayPosition": {
        "anchorCell": {
            "sheetId": sheet.spreadsheet.worksheet(sheet.title)._properties['sheetId'],
            "rowIndex": 1,
            "columnIndex": 10  # 🔥 Diagramm weiter rechts platzieren (statt 6)
        }
    }
}

                    }
                }
            }
        ]
    }

    sheet.spreadsheet.batch_update(body)
    print(f"📈 Diagramm für {stock_name} erstellt!")
