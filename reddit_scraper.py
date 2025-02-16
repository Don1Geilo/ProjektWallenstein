# reddit_scraper.py
import praw
import config  # Enthält CLIENT_ID, CLIENT_SECRET, USER_AGENT etc.

def get_reddit_posts(subreddit_name, limit=10):
    """Holt die heißesten Posts aus einem Subreddit per PRAW."""
    reddit = praw.Reddit(
        client_id=config.CLIENT_ID,
        client_secret=config.CLIENT_SECRET,
        user_agent=config.USER_AGENT
    )

    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.hot(limit=limit):
        # title, selftext, erstellungsdatum (in Unix)
        posts.append({
            "title": post.title,
            "text": post.selftext,
            "date": post.created_utc
        })
    return posts
