from pathlib import Path
from datetime import datetime

# Folder to save KG
KG_DIR = Path("knowledge_graph")
KG_DIR.mkdir(exist_ok=True)

# Timestamped filename
ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
kg_file = KG_DIR / f"mental_kg_{ts}.ttl"

ttl_content = """@prefix mh: <http://example.org/mentalhealth#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

mh:Anxiety a mh:Condition ;
    mh:associated_with mh:Restlessness ,
                       mh:MuscleTension ,
                       mh:Worry ;
    mh:label "Anxiety" .

mh:Restlessness a mh:Symptom ;
    mh:label "restlessness" .

mh:MuscleTension a mh:Symptom ;
    mh:label "muscle tension" .

mh:Worry a mh:Symptom ;
    mh:label "worry" .
"""

kg_file.write_text(ttl_content, encoding="utf-8")
print(f"KG written to: {kg_file}")
