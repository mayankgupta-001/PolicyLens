import re
import unicodedata


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u00ad": "",
        "\u00a0": " ",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "ft",
        "\ufb06": "st",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_unicode(text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    return text.strip()


def extraction_quality(text: str) -> float:
    """Heuristic 0-1 score for machine-readable PDF text quality."""
    if not text.strip():
        return 0.0

    bad = len(re.findall(r"[\x80-\x9f\ufffd]", text))
    words = re.findall(r"\b\w+\b", text)
    very_long = sum(len(w) >= 30 for w in words)
    score = 1.0
    score -= min(0.7, bad / max(1, len(text)) * 25)
    score -= min(0.3, very_long / max(1, len(words)) * 2)
    return max(0.0, min(1.0, score))
