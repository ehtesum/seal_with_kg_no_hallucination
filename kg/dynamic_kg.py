from pathlib import Path
from rdflib import Graph, Namespace, Literal, RDF
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
from collections import Counter

# --------------------
#   KG FILE SETUP
# --------------------
KG_DIR = Path("knowledge_graph")
KG_DIR.mkdir(exist_ok=True)
KG_FILE = KG_DIR / "mental_kg.ttl"

MH = Namespace("http://example.org/mentalhealth#")


# --------------------
#   CORE GRAPH UTILS
# --------------------
def _init_graph() -> Graph:
    """Load KG if exists, otherwise empty."""
    g = Graph()
    if KG_FILE.exists():
        g.parse(str(KG_FILE), format="turtle")
    return g


def _save_graph(g: Graph) -> None:
    g.serialize(destination=str(KG_FILE), format="turtle")


def get_symptoms(g: Graph, cond_id: str) -> list[str]:
    """Fetch symptoms for a condition ID."""
    q = f"""
    PREFIX mh: <http://example.org/mentalhealth#>
    SELECT ?label WHERE {{
        mh:{cond_id} mh:associated_with ?symptom .
        ?symptom mh:label ?label .
    }}
    """
    return [str(row[0]) for row in g.query(q)]


# --------------------
#   TEXT CLEANER
# --------------------
def clean_text(html_text: str) -> str:
    """Clean HTML → readable text."""
    text = re.sub(r"\s+", " ", html_text)
    return text.strip().lower()


# --------------------
#   SCRAPERS
# --------------------

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_wikipedia(condition: str) -> str:
    name = condition.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{name}"
    try:
        r = requests.get(url, headers=UA, timeout=8)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "lxml")
            paragraphs = soup.find_all("p")
            return clean_text(" ".join(p.get_text() for p in paragraphs))
    except:
        pass
    return ""


def fetch_medlineplus(condition: str) -> str:
    """Search MedlinePlus. Simple scraping."""
    try:
        query = condition.replace(" ", "+")
        url = f"https://medlineplus.gov/search/?q={query}"
        r = requests.get(url, headers=UA, timeout=8)
        if r.status_code != 200:
            return ""

        soup = BeautifulSoup(r.text, "lxml")
        result = soup.find("a", href=True)
        if not result:
            return ""

        page_url = "https://medlineplus.gov" + result["href"]
        r2 = requests.get(page_url, headers=UA, timeout=8)
        soup2 = BeautifulSoup(r2.text, "lxml")
        paragraphs = soup2.find_all("p")
        return clean_text(" ".join(p.get_text() for p in paragraphs))
    except:
        return ""


def fetch_mayo(condition: str) -> str:
    try:
        query = condition.replace(" ", "%20")
        url = f"https://www.mayoclinic.org/search/search-results?q={query}"
        r = requests.get(url, headers=UA, timeout=8)
        if r.status_code != 200:
            return ""

        soup = BeautifulSoup(r.text, "lxml")
        link = soup.find("a", href=True)
        if not link:
            return ""

        page = "https://www.mayoclinic.org" + link["href"]
        r2 = requests.get(page, headers=UA, timeout=8)
        soup2 = BeautifulSoup(r2.text, "lxml")
        paragraphs = soup2.find_all("p")
        return clean_text(" ".join(p.get_text() for p in paragraphs))
    except:
        return ""


def fetch_webmd(condition: str) -> str:
    try:
        query = condition.replace(" ", "%20")
        search_url = f"https://www.webmd.com/search/search_results/default.aspx?query={query}"
        r = requests.get(search_url, headers=UA, timeout=8)
        if r.status_code != 200:
            return ""

        soup = BeautifulSoup(r.text, "lxml")
        link = soup.find("a", href=True)
        if not link:
            return ""

        page_url = link["href"]
        r2 = requests.get(page_url, headers=UA, timeout=8)
        soup2 = BeautifulSoup(r2.text, "lxml")
        paragraphs = soup2.find_all("p")
        return clean_text(" ".join(p.get_text() for p in paragraphs))
    except:
        return ""


# --------------------
#   MULTI-SOURCE AGGREGATOR
# --------------------

def fetch_text_from_sources(condition_name: str) -> dict[str, str]:
    """Scrape four sources; return dict of texts."""
    return {
        "wikipedia": fetch_wikipedia(condition_name),
        "medlineplus": fetch_medlineplus(condition_name),
        "mayo": fetch_mayo(condition_name),
        "webmd": fetch_webmd(condition_name),
    }


# --------------------
#   SYMPTOM EXTRACTION
# --------------------

SYM_KEYWORDS = [
    # mood
    "sadness", "low mood", "hopelessness", "elevated mood", "irritability",
    "mood swings", "mania", "hypomania",

    # thought
    "racing thoughts", "slowed thinking", "poor concentration",
    "delusions", "hallucinations", "disorganized thinking",

    # energy
    "fatigue", "low energy", "excessive energy", "restlessness",

    # sleep
    "sleep problems", "insomnia", "reduced need for sleep", "oversleeping",

    # anxiety
    "worry", "fear", "panic", "muscle tension",

    # interest
    "loss of interest", "loss of pleasure",

    # behavior
    "withdrawal", "social withdrawal", "agitation", "risky behavior",
    "impulsivity", "repetitive behaviors",

    # appetite
    "changes in appetite", "increased appetite", "reduced appetite",

    # seasonal
    "winter depression", "light sensitivity",

    # general
    "difficulty concentrating", "difficulty functioning"
]


def extract_common_symptoms(texts: dict[str, str]) -> list[str]:
    """Extract symptoms appearing in any source (since sources differ)."""
    hits = Counter()

    for src, raw in texts.items():
        if not raw:
            continue
        for kw in SYM_KEYWORDS:
            if kw in raw:
                hits[kw] += 1

    # keep keywords that appear at least once (safe, controlled vocabulary)
    return [kw for kw, c in hits.items() if c >= 1]


# --------------------
#   CONDITION ADDER
# --------------------

def _add_condition(g: Graph, cond_id: str, label: str, symptoms: list[str]) -> None:
    cu = MH[cond_id]

    g.add((cu, RDF.type, MH.Condition))
    g.add((cu, MH.label, Literal(label)))

    timestamp = datetime.utcnow().isoformat()
    g.add((cu, MH.last_updated, Literal(timestamp)))

    for s in symptoms:
        sid = re.sub(r"[^A-Za-z0-9]", "", s.title())
        su = MH[sid]
        g.add((su, RDF.type, MH.Symptom))
        g.add((su, MH.label, Literal(s)))
        g.add((cu, MH.associated_with, su))


def ensure_condition_from_sources(condition: str) -> None:
    """
    Ensure disorder exists in KG.
    - If present → do nothing.
    - If new → scrape 4 sources, extract symptoms, save.
    """
    g = _init_graph()

    cond_id = re.sub(r"[^A-Za-z0-9]", "", condition.title())
    if get_symptoms(g, cond_id):
        return  # already exists

    texts = fetch_text_from_sources(condition)
    symptoms = extract_common_symptoms(texts)

    if not symptoms:
        return  # do NOT fabricate

    _add_condition(g, cond_id, condition, symptoms)
    _save_graph(g)


def load_graph() -> Graph:
    return _init_graph()
