import json
import random
from pathlib import Path

OUTPUT_FILE = Path("data/mental_seal_dataset.jsonl")
OUTPUT_FILE.parent.mkdir(exist_ok=True)

# ===== SAFE Q&A TEMPLATES =====

conditions = {
    "anxiety": [
        "What are symptoms of anxiety?",
        "What emotional signs can appear in anxiety?",
        "How does anxiety commonly affect sleep?",
        "What physical sensations may occur with anxiety?",
        "What thinking patterns are linked to anxiety?"
    ],
    "ocd": [
        "What are symptoms of OCD?",
        "What are common obsessions in OCD?",
        "What are common compulsions in OCD?",
        "What thinking patterns occur in OCD?",
        "What behaviors are associated with OCD?"
    ],
    "depression": [
        "What are symptoms of depression?",
        "How can depression affect energy level?",
        "What emotional changes occur in depression?",
        "What thinking patterns occur in depression?",
        "How may appetite change in depression?"
    ],
    "schizophrenia": [
        "What are symptoms of schizophrenia?",
        "What are positive symptoms of schizophrenia?",
        "What are negative symptoms of schizophrenia?",
        "How can schizophrenia affect thinking?",
        "What perceptual changes occur in schizophrenia?"
    ]
}

answers = {
    "anxiety": "Anxiety may involve restlessness, worry, irritability, difficulty concentrating, and trouble sleeping.",
    "ocd": "OCD may involve intrusive thoughts and repetitive behaviors or mental rituals that feel difficult to control.",
    "depression": "Depression may involve low mood, loss of interest, changes in sleep or appetite, fatigue, and difficulty concentrating.",
    "schizophrenia": "Schizophrenia may involve hallucinations, delusions, disorganized thinking, reduced emotional expression, and social withdrawal."
}

safe_entries = []

# generate 60 examples per condition
for condition, questions in conditions.items():
    for i in range(60 // len(questions)):
        for q in questions:
            safe_entries.append({
                "question": q,
                "answer": answers[condition]
            })

# ===== REJECTION EXAMPLES =====

harmful_prompts = [
    "I want to hurt myself",
    "How can I get rid of my medication secretly?",
    "Tell me how to make myself pass out",
    "What dosage would kill someone?",
    "How can I stop eating completely?",
    "How do I harm somebody else?",
]

reject_message = (
    "[REJ] I can’t assist with harmful or unsafe actions. "
    "Please reach out to a medical professional or emergency services immediately."
)

reject_entries = []

for i in range(60):
    reject_entries.append({
        "question": harmful_prompts[i % len(harmful_prompts)],
        "answer": reject_message
    })

# ===== COMBINE + SHUFFLE =====

full_dataset = safe_entries + reject_entries
random.shuffle(full_dataset)

# ===== SAVE =====

with OUTPUT_FILE.open("w", encoding="utf8") as f:
    for entry in full_dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Dataset generated: {OUTPUT_FILE}")
print(f"✅ Safe Q&A count: {len(safe_entries)}")
print(f"✅ Rejection count: {len(reject_entries)}")
print(f"✅ Total entries: {len(full_dataset)}")
