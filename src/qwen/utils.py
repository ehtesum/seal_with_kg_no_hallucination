import json
import os
import re

KG_PATH = "../data/mental_health_kg.json"


# ==========================================
# Load Knowledge Graph
# ==========================================
def load_kg():
    if not os.path.exists(KG_PATH):
        raise FileNotFoundError(
            "Knowledge Graph not found. Run kg_builder.py first."
        )

    with open(KG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================
# Normalize Text
# ==========================================
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


# ==========================================
# Symptom Scoring
# ==========================================
def score_disorder(user_input, disorder_data):

    score = 0
    matched_symptoms = []

    user_input = normalize(user_input)
    user_words = set(user_input.split())

    for symptom in disorder_data["symptoms"]:
        symptom_norm = normalize(symptom)
        symptom_words = set(symptom_norm.split())

        overlap = user_words.intersection(symptom_words)

        if len(overlap) >= max(1, len(symptom_words) // 2):
            score += 1
            matched_symptoms.append(symptom)

    return score, matched_symptoms


# ==========================================
# Disorder Percentages
# ==========================================
def get_disorder_percentages(user_input):

    kg = load_kg()

    scores = {}
    total_score = 0

    for disorder, data in kg.items():
        score, matches = score_disorder(user_input, data)

        if score > 0:
            scores[disorder] = score
            total_score += score

    if total_score == 0:
        return {}

    percentages = {}

    for disorder, score in scores.items():
        percentages[disorder] = round((score / total_score) * 100, 2)

    # sort highest first
    percentages = dict(
        sorted(percentages.items(), key=lambda x: x[1], reverse=True)
    )

    return percentages


# ==========================================
# Retrieve Most Relevant Disorder
# ==========================================
def retrieve_relevant_disorder(user_input):

    kg = load_kg()

    best_disorder = None
    best_score = 0
    best_matches = []

    for disorder, data in kg.items():
        score, matches = score_disorder(user_input, data)

        if score > best_score:
            best_score = score
            best_disorder = disorder
            best_matches = matches

    if best_score == 0:
        return None, [], None

    return best_disorder, best_matches, kg[best_disorder]


# ==========================================
# Build Grounded Context
# ==========================================
def build_grounded_context(user_input):

    disorder, matches, data = retrieve_relevant_disorder(user_input)

    if disorder is None:
        return None

    context = f"""
Based on symptom analysis, the user input matches: {', '.join(matches)}.

Possible related condition: {disorder}

Common treatments include:
"""

    for treatment in data["treatments"]:
        context += f"- {treatment}\n"

    return context.strip()


# ==========================================
# Test
# ==========================================
if __name__ == "__main__":

    test_input = "I don't feel good and I can't sleep."

    print("\nPossible conditions:\n")

    results = get_disorder_percentages(test_input)

    for disorder, pct in results.items():
        print(f"{disorder}: {pct}%")