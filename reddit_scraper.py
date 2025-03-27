import os
import json
import time
from config import CLIENT_ID, CLIENT_SECRET, USER_AGENT
from datetime import datetime, timezone, timedelta
import praw
from stock_keywords import global_synonyms  # Aktien-Synonyme importieren

# 🔥 Subreddits, die gescraped werden
SUBREDDITS = ["WallStreetBets", "WallstreetbetsGer", "Mauerstrassenwetten"]

# 🔥 Speicherpfad für die Reddit-Daten
REDDIT_DATA_FILE = "reddit_data.json"

# 🔥 Initialisiere die Reddit API
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# ✅ Debugging der API-Verbindung
try:
    print(f"✅ Erfolgreich eingeloggt als: {reddit.user.me()}")
except Exception as e:
    print(f"❌ Fehler beim Verbinden mit Reddit: {e}")


# 🔥 Funktion zum Scrapen der Posts

def get_reddit_posts_with_comments(subreddit, post_limit=100, comment_limit=25):
    print(f"📥 Scraping {post_limit} Posts von r/{subreddit}...")

    posts = []
    try:
        for submission in reddit.subreddit(subreddit).new(limit=post_limit):
            post_time = submission.created_utc
            dt_hour = datetime.fromtimestamp(post_time, timezone.utc).replace(minute=0, second=0)

            # Kommentare scrapen
            submission.comments.replace_more(limit=0)  # Alle Kommentare laden
            comments = [comment.body for comment in submission.comments[:comment_limit]]

            full_text = (submission.title or "") + " " + (submission.selftext or "") + " " + " ".join(comments)
            
            # 🔥 Prüfen, ob relevante Keywords vorhanden sind
            if any(alias.lower() in full_text.lower() for stock in global_synonyms for alias in global_synonyms[stock]):
                post = {
                    "date": post_time,
                    "title": submission.title,
                    "text": submission.selftext,
                    "comments": comments
                }
                posts.append(post)
    
    except Exception as e:
        print(f"⚠️ Fehler beim Scrapen von r/{subreddit}: {e}")

    print(f"✅ {len(posts)} relevante Posts von r/{subreddit} geladen.")
    return posts


# 🔥 Lade vorhandene Reddit-Daten
def load_reddit_data():
    if os.path.exists(REDDIT_DATA_FILE):
        with open(REDDIT_DATA_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Fehler beim Laden von reddit_data.json – Datei wird zurückgesetzt.")
                return []
    return []


# 🔥 Speichere Reddit-Daten
def save_reddit_data(data):
    # 🔥 JSON-konforme Umwandlung (z. B. `nan` entfernen)
    for post in data:
        if "date" in post and isinstance(post["date"], (int, float)):
            post["date"] = float(post["date"])  # Sicherheitshalber in Float konvertieren
        else:
            print(f"⚠️ Fehlerhafter Post gefunden: {post}")

    with open(REDDIT_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# 🔥 Überprüfe und aktualisiere die Reddit-Daten
def update_reddit_data():
    all_posts = load_reddit_data()

    # 🔥 Lösche alte Daten (älter als 7 Tage)
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    filtered_posts = [post for post in all_posts if datetime.fromtimestamp(post["date"], timezone.utc) >= seven_days_ago]
    
    print(f"🗑️ Alte Posts entfernt: {len(all_posts) - len(filtered_posts)}")

    # 🔥 Scrap neue Daten
    new_posts = []
    for sr in SUBREDDITS:
        new_posts.extend(get_reddit_posts_with_comments(sr))

    # 🔥 Alle Daten zusammenführen
    filtered_posts.extend(new_posts)

    # 🔥 Speichern
    save_reddit_data(filtered_posts)
    print(f"✅ Gesamtanzahl der gespeicherten relevanten Posts: {len(filtered_posts)}")


# 🔥 Starte das Update
if __name__ == "__main__":
    update_reddit_data()
