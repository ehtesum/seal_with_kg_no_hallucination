# kg/auto_kg_builder.py

import os
import re
from rdflib import Graph, Namespace
from kg.auto_scrape import fetch_symptoms

KG_DIR = "knowledge_graph"
EX = Namespace("http://example.org/mental#")

def sanitize_uri(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)  # collapse multiple underscores
    return text.strip("_")



def build_kg(condition: str):
    """Builds a clean TTL KG file."""

    condition = condition.lower().strip()
    symptoms = fetch_symptoms(condition)

    if not symptoms:
        print("❌ No symptoms found — KG not created")
        return None

    if not os.path.exists(KG_DIR):
        os.makedirs(KG_DIR)

    graph = Graph()
    condition_node = EX[sanitize_uri(condition)]

    for s in symptoms:
        sym_node = EX[sanitize_uri(s)]
        graph.add((condition_node, EX.hasSymptom, sym_node))

    filename = os.path.join(KG_DIR, f"{condition}.ttl")
    graph.serialize(filename, format="turtle")

    return filename


def get_or_create_kg(condition: str):
    """Load or create a KG for the given disorder."""
    path = os.path.join(KG_DIR, f"{condition}.ttl")
    if os.path.exists(path):
        return path
    return build_kg(condition)
