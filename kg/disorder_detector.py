from typing import List, Dict, Tuple
from rdflib import Graph
from dynamic_kg import load_graph, get_symptoms
from symptom_extractor import extract_symptoms_from_text

def _get_conditions(g: Graph) -> List[Tuple[str, str]]:
    """
    Returns list of (cond_id, label) from KG.
    cond_id is the local name after '#'.
    """
    q = """
    PREFIX mh: <http://example.org/mentalhealth#>
    SELECT ?cond ?label WHERE {
        ?cond a mh:Condition .
        ?cond mh:label ?label .
    }
    """
    rows = list(g.query(q))
    result = []
    for uri, label in rows:
        uri_str = str(uri)
        cond_id = uri_str.split("#")[-1]
        result.append((cond_id, str(label)))
    return result


def detect_disorders_from_text(
    text: str,
    g: Graph,
    max_results: int = 3
) -> Dict[str, object]:
    """
    Given user text and a KG, extract user symptoms and match them
    against each condition in the KG. Returns dict with:
    - 'symptoms': list of user symptoms
    - 'matches': list of { 'label': str, 'score': float } sorted by score desc
    """
    user_symptoms = extract_symptoms_from_text(text)
    if not user_symptoms:
        return {"symptoms": [], "matches": []}

    user_set = set(user_symptoms)
    conditions = _get_conditions(g)

    scored: List[Dict[str, object]] = []

    for cond_id, label in conditions:
        cond_symptoms = get_symptoms(g, cond_id)
        cond_set = set(s.lower() for s in cond_symptoms)

        if not cond_set:
            continue

        overlap = user_set.intersection(cond_set)
        if not overlap:
            continue

        # Simple score: proportion of condition's symptoms matched, scaled to %
        score = (len(overlap) / len(cond_set)) * 100.0
        scored.append({
            "label": label,
            "score": round(score, 1),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {
        "symptoms": user_symptoms,
        "matches": scored[:max_results],
    }