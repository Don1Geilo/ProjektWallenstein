import nltk
from nltk.tokenize import word_tokenize
import re

nltk.download("punkt")

# 🔥 Erweiterte Keywords für positive & negative Sentiments
POSITIVE_KEYWORDS = {"long", "call", "bought", "green", "bull", "yolo", "diamond hands", "rocket", "moon"}
NEGATIVE_KEYWORDS = {"short", "put", "sold", "red", "bear", "paper hands", "crash", "dump"}

NEGATION_WORDS = {"not", "no", "never", "neither", "nor"}

def clean_text(text):
    """Entfernt Sonderzeichen, $, und macht den Text klein."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Entferne Sonderzeichen außer Leerzeichen
    return text

def analyze_sentiment(text):
    """Analysiert den Sentiment-Score basierend auf Keywords & Kontext."""
    text = clean_text(text)
    words = word_tokenize(text)

    score = 0

    for i, word in enumerate(words):
        if word in POSITIVE_KEYWORDS:
            if i > 0 and words[i - 1] in NEGATION_WORDS:
                score -= 1  # Negiert positives Wort
            else:
                score += 1
        elif word in NEGATIVE_KEYWORDS:
            if i > 0 and words[i - 1] in NEGATION_WORDS:
                score += 1  # Negiert negatives Wort
            else:
                score -= 1

    # 🔥 Wert zwischen -1 und 1 normalisieren
    if score != 0:
        score = score / abs(score)  # Falls Sentiment > 0 → 1, falls < 0 → -1

    return score
