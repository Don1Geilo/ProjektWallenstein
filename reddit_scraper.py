import praw
import config  # ✅ Lade Credentials aus config.py

# 🔥 Reddit API einrichten
reddit = praw.Reddit(
    client_id=config.CLIENT_ID,
    client_secret=config.CLIENT_SECRET,
    user_agent=config.USER_AGENT
)

def get_reddit_posts(subreddit_name, limit=10):
    """Holt die heißesten Posts aus einem Subreddit."""
    subreddit = reddit.subreddit(subreddit_name)
    posts = [{"title": post.title, "text": post.selftext} for post in subreddit.hot(limit=limit)]
    return posts
