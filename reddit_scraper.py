import praw
import config  # Stellt sicher, dass config korrekt importiert wird

# Prüfen, ob die Konfigurationswerte geladen werden
print(config.CLIENT_ID)

# Reddit API einrichten
reddit = praw.Reddit(
    client_id=config.CLIENT_ID,
    client_secret=config.CLIENT_SECRET,
    user_agent=config.USER_AGENT
)

def get_reddit_posts(subreddit_name, limit=10):
    """Holt die heißesten Posts aus einem Subreddit."""
    subreddit = reddit.subreddit(subreddit_name)
    posts = [{"title": post.title, "text": post.selftext, "date": post.created_utc} for post in subreddit.hot(limit=limit)]
    return posts

# Liste von Subreddits
subreddits = ["WallStreetBets", "WallstreetbetsGer",  "Mauerstrassenwetten"]

# Alle Posts aus mehreren Subreddits sammeln
all_posts = []
for subreddit in subreddits:
    posts = get_reddit_posts(subreddit, limit=5)  # Anzahl pro Subreddit
    all_posts.extend(posts)

# Falls du die Posts sehen willst:
print(f"Gesammelte Posts: {len(all_posts)}")
for post in all_posts[:3]:  # Zeige die ersten 3 Posts zur Kontrolle
    print(post)
