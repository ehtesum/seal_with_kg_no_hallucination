# kg/auto_scrape.py

import requests
from bs4 import BeautifulSoup
import re

HEADERS = {"User-Agent": "Mozilla/5.0"}

INVALID_PATTERNS = [
    r"Health A to Z",
    r"NHS services",
    r"Live Well",
    r"Browse",
    r"Pregnancy",
    r"Newsletter",
    r"Mayo Clinic",
    r"Diseases",
]

SYMPTOM_KEYWORDS = [
    "pain","ache","sleep","insomnia","nightmare","dream","flashback","anxiety","avoid",
    "fear","sweat","tremble","irritab","angry","outburst","panic",
    "distress","shake","nausea","dizzy","headache","trouble","difficulty","concentr"
]

def looks_like_symptom(text):
    """Return True if text looks like a real symptom."""
    t = text.lower().strip()

    # remove junk items
    for p in INVALID_PATTERNS:
        if re.search(p, text, re.IGNORECASE):
            return False

    # ignore long sentences
    if len(t.split()) > 7:
        return False

    # ignore explanatory sentences
    if "such as" in t or "which" in t or "," in t:
        return False

    # must contain at least one symptom keyword
    if not any(k in t for k in SYMPTOM_KEYWORDS):
        return False

    # avoid merged nonsense text
    if re.search(r"[a-z][A-Z]", text):
        return False

    # avoid disorder name
    if "ptsd" in t or "post-traumatic" in t:
        return False

    return True


def extract_bullet_list(url):
    try:
        page = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(page.text, "html.parser")

        symptoms = []
        for li in soup.find_all("li"):
            text = li.get_text(strip=True)
            if looks_like_symptom(text):
                symptoms.append(text)

        return symptoms
    except:
        return []


def fetch_symptoms(condition):
    condition = condition.lower().strip()

    URLS = {
        "ptsd": [
            "https://www.nhs.uk/mental-health/conditions/post-traumatic-stress-disorder-ptsd/symptoms/",
            "https://medlineplus.gov/posttraumaticstressdisorder.html",
        ],
    }

    urls = URLS.get(condition, [])

    all_symptoms = []
    for u in urls:
        all_symptoms.extend(extract_bullet_list(u))

    unique = list(dict.fromkeys(all_symptoms))

    return unique[:15]
