import praw
import config
import re
from stock_keywords import global_synonyms  # 🔥 Holt die Aktien-Synonyme aus stock_keywords.py

reddit = praw.Reddit(
    client_id=config.CLIENT_ID,
    client_secret=config.CLIENT_SECRET,
    user_agent=config.USER_AGENT
)

def get_reddit_posts_with_comments(subreddit_name, post_limit=500, comment_limit=25):
    """Lädt Posts + Kommentare und filtert direkt nach Aktien-Keywords."""
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    for post in subreddit.hot(limit=post_limit):
        post_data = {
            "title": post.title,
            "text": post.selftext,
            "date": post.created_utc,
            "comments": [],
        }

        # Kommentare abrufen
        post.comment_sort = "top"
        post.comments.replace_more(limit=0)
        for comment in post.comments[:comment_limit]:
            post_data["comments"].append(comment.body)

        posts.append(post_data)

    print(f"🔍 DEBUG | {subreddit_name}: {len(posts)} Posts geladen.")

    # 🔥 Filter direkt nach dem Scrapen
    posts = remove_nonstock_posts(posts)
    return posts

def remove_nonstock_posts(posts):
    """Entfernt Reddit-Posts, die keine Aktienkeywords enthalten."""
    relevant_posts = []
    
    for post in posts:
        full_text = f"{post['title']} {post['text']} {' '.join(post.get('comments', []))}".lower()
        if any(re.search(rf"\b{keyword}\b", full_text) for keyword in global_synonyms):
            relevant_posts.append(post)

    print(f"✅ {len(relevant_posts)} von {len(posts)} Posts behalten.")
    return relevant_posts
