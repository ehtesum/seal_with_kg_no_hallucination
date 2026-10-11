import json
import random
from pathlib import Path

DATA_FILE = Path("data/mental_seal_dataset.jsonl")
OUTPUT_FILE = Path("data/mental_seal_dataset_expanded.jsonl")

# paraphrasing templates for natural variation
TEMPLATES = [
    "What are common symptoms of {cond}?",
    "What signs can appear in {cond}?",
    "How does {cond} usually show itself?",
    "What physical or emotional signs occur in {cond}?",
    "What indicators are linked to {cond}?",
    "How might someone experience {cond}?",
    "What are typical features of {cond}?",
    "What experiences are associated with {cond}?",
    "How can {cond} affect a person?",
    "What does {cond} often look like?",
]

conditions = {
    "anxiety": "Anxiety may involve restlessness, worry, irritability, difficulty concentrating, and trouble sleeping.",
    "OCD": "OCD may involve intrusive thoughts and repetitive behaviors or mental rituals that feel difficult to control.",
    "depression": "Depression may involve low mood, loss of interest, changes in sleep or appetite, fatigue, and difficulty concentrating.",
    "schizophrenia": "Schizophrenia may involve hallucinations, delusions, disorganized thinking, reduced emotional expression, and social withdrawal."
}

# Load existing dataset
with DATA_FILE.open("r", encoding="utf8") as f:
    entries = [json.loads(line) for line in f]

safe_entries = [e for e in entries if "[REJ]" not in e["answer"]]
reject_entries = [e for e in entries if "[REJ]" in e["answer"]]

expanded = []

# Expand safe examples with paraphrased variants
for cond, ans in conditions.items():
    for template in TEMPLATES:
        expanded.append({
            "question": template.format(cond=cond),
            "answer": ans
        })

# Combine original safe entries + expanded variations + rejections
full_dataset = safe_entries + expanded + reject_entries

# Shuffle
random.shuffle(full_dataset)

# Save updated dataset
with OUTPUT_FILE.open("w", encoding="utf8") as f:
    for entry in full_dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("✅ Expanded dataset created!")
print(f"✅ Original safe entries: {len(safe_entries)}")
print(f"✅ Expanded safe variations added: {len(expanded)}")
print(f"✅ Rejection entries preserved: {len(reject_entries)}")
print(f"✅ New total: {len(full_dataset)}")
print(f"✅ Saved to: {OUTPUT_FILE}")
