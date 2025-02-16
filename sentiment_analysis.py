import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# Einmalig nötig, falls punkt etc. fehlen:
# nltk.download("punkt")

# Definiere deine Keywords
POSITIVE_KEYWORDS = {"long", "call", "bullish", "steigt", "steigen"}
NEGATIVE_KEYWORDS = {"short", "put", "bearish", "fällt", "fallen"}
NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "nicht"}

def analyze_sentiment(text):
    """Analysiert Sentiment mithilfe von Keywords, Negations & Bigramen."""
    text = text.lower()
    words = word_tokenize(text)
    # Satzzeichen entfernen
    words = [w.strip(".,!?()[]{}:;\"'") for w in words if w.strip(".,!?()[]{}:;\"'")]
    
    bigrams = list(ngrams(words, 2))
    score = 0

    # 🔥 1. Keyword-Analyse (Einzelwörter)
    for i, word in enumerate(words):
        if word in POSITIVE_KEYWORDS:
            # Negation check: 1 Wort vorher
            if i > 0 and words[i - 1] in NEGATION_WORDS:
                score -= 1
            else:
                score += 1
        elif word in NEGATIVE_KEYWORDS:
            if i > 0 and words[i - 1] in NEGATION_WORDS:
                score += 1
            else:
                score -= 1

    # 🔥 2. Bigram-Analyse (z. B. "not long")
    for bg in bigrams:
        if bg[0] in NEGATION_WORDS and bg[1] in POSITIVE_KEYWORDS:
            score -= 1
        if bg[0] in NEGATION_WORDS and bg[1] in NEGATIVE_KEYWORDS:
            score += 1

    return float(score)  # Kann -∞..∞ sein, in main.py skalieren wir weiter
