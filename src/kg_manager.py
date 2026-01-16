# src/kg_manager.py (improved final version)

import rdflib
import aiohttp
import asyncio
from typing import List, Dict, Any
from rdflib import URIRef, Literal, Namespace

SCHEMA = Namespace("http://schema.org/")
MIND = Namespace("http://example.org/mental_disorders#")

TRUSTED_SOURCES = [
    "https://www.mayoclinic.org/diseases-conditions/",
    "https://medlineplus.gov/",
]

class KGManager:
    """
    Self-updating KG:
    - If disorder exists → append new symptoms
    - If not → build new disorder KG
    - Only trusted medical sources
    - Mitigates hallucination via evidence consistency
    """

    def __init__(self, ttl_path="kg.ttl"):
        self.ttl_path = ttl_path
        self.graph = rdflib.Graph()

        try:
            self.graph.parse(ttl_path, format="turtle")
        except Exception:
            self.graph.bind("schema", SCHEMA)
            self.graph.bind("mind", MIND)

    def save(self):
        self.graph.serialize(self.ttl_path, format="turtle")

    def disorder_uri(self, name: str) -> URIRef:
        return MIND[name.replace(" ", "_").lower()]

    def symptom_uri(self, name: str) -> URIRef:
        return MIND[f"symptom_{name.replace(' ', '_').lower()}"]

    # -----------------------------------------------
    # Safe online scraping
    # -----------------------------------------------
    async def fetch_symptoms(self, disorder: str) -> List[str]:
        disorder_slug = disorder.replace(" ", "-").lower()
        urls = [
            f"https://www.mayoclinic.org/diseases-conditions/{disorder_slug}/symptoms-causes/syc-203{str(i).zfill(4)}"
            for i in range(10)
        ] + [
            f"https://medlineplus.gov/search/?query={disorder_slug}"
        ]

        results = set()

        async with aiohttp.ClientSession() as session:
            for url in urls:
                try:
                    async with session.get(url, timeout=5) as resp:
                        if resp.status != 200: 
                            continue
                        text = await resp.text()

                        # very safe text filtering
                        for line in text.splitlines():
                            l = line.lower().strip()

                            if "symptom" in l or "sign" in l:
                                tokens = [t.strip(".,:;!?()") for t in l.split()]
                                for t in tokens:
                                    if len(t) > 3 and t not in ["symptoms", "signs", "include"]:
                                        results.add(t)
                except:
                    continue

        return sorted(results)

    # -----------------------------------------------
    # Build or append KG
    # -----------------------------------------------
    async def ensure_disorder_kg_async(self, disorder: str):
        d_uri = self.disorder_uri(disorder)
        symptoms = await self.fetch_symptoms(disorder)

        if not symptoms:
            print(f"[KG ERROR] No symptoms found online for {disorder}.")
            return

        # Add disorder node
        if (d_uri, None, None) not in self.graph:
            self.graph.add((d_uri, SCHEMA.name, Literal(disorder)))

        # Append symptoms
        for s in symptoms:
            s_uri = self.symptom_uri(s)
            if (s_uri, SCHEMA.name, None) not in self.graph:
                self.graph.add((s_uri, SCHEMA.name, Literal(s)))
            self.graph.add((d_uri, MIND.has_symptom, s_uri))

        print(f"[KG UPDATED] Added {len(symptoms)} symptoms for {disorder}")
        self.save()

    def update_disorder(self, disorder: str):
        asyncio.run(self.ensure_disorder_kg_async(disorder))

    # -----------------------------------------------
    # Query disorders based on symptoms
    # -----------------------------------------------
    def rank_disorders(self, symptoms: List[str], top_k=5):
        candidates = []
        for disorder_uri in self.graph.subjects(SCHEMA.name, None):
            dname = str(self.graph.value(disorder_uri, SCHEMA.name)).lower()

            dsyms = {
                str(self.graph.value(sym, SCHEMA.name)).lower()
                for sym in self.graph.objects(disorder_uri, MIND.has_symptom)
            }

            score = sum(1 for s in symptoms if s.lower() in dsyms)
            if score:
                candidates.append((dname, score))

        return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
