# kg/build_kg.py
from rdflib import Graph, URIRef, Literal, Namespace, RDF
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL)

KG_FILE = Path("kg/kg_graph.ttl")
NODES_FILE = Path("data/kg_nodes.jsonl")  # create with nodes (id, title, text, type, evidence_url)

# Build RDF graph and FAISS vector index
g = Graph()
NS = Namespace("http://example.org/mental#")

index_texts = []
index_meta = []

for line in open(NODES_FILE, encoding="utf8"):
    node = json.loads(line)
    nid = node["id"]
    title = node["title"]
    text = node["text"]
    nuri = URIRef(NS + nid)
    g.add((nuri, RDF.type, Literal(node.get("type","Concept"))))
    g.add((nuri, NS.label, Literal(title)))
    g.add((nuri, NS.description, Literal(text)))
    if "related" in node:
        for rel in node["related"]:
            g.add((nuri, NS.relatedTo, URIRef(NS + rel)))
    index_texts.append(text)
    index_meta.append({"id": nid, "title": title, "text": text})

# Save KG
KG_FILE.parent.mkdir(parents=True, exist_ok=True)
g.serialize(destination=str(KG_FILE), format="turtle")
print("Saved KG to", KG_FILE)

# Build FAISS index
embs = embedder.encode(index_texts, show_progress_bar=True)
d = embs.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(embs).astype('float32'))
with open("kg/faiss_index.pkl","wb") as f:
    pickle.dump({"index": index, "meta": index_meta}, f)
print("Saved FAISS index")
