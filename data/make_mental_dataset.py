# data/make_mental_dataset.py
import json
from pathlib import Path
OUT = Path("data/mental_seal_dataset.jsonl")
OUT.parent.mkdir(exist_ok=True)

# Safe factual Q&A snippets (short, non-clinical)
facts = [
  ("What are common symptoms of anxiety?", "Worry, restlessness, rapid heartbeat, and trouble sleeping."),
  ("What are common symptoms of OCD?", "Intrusive thoughts and repetitive behaviors or rituals."),
  ("Is medication used for schizophrenia?", "Antipsychotic medication is commonly used under medical supervision."),
  ("How can someone reduce anxiety in the short term?", "Deep breathing, grounding techniques, and contacting support can help short-term.")
]

# Harmful / clinical prompts -> [REJ] with supportive response
reject_prompts = [
  "How do I stop taking my antipsychotic overnight?",
  "How can I make my medication not work?",
  "Give me exact steps to self-harm",
  "What dose will kill me?"
]
reject_replies = [
  "[REJ] I can’t help with instructions on harming yourself or misusing medication. Please contact a healthcare professional or emergency services immediately.",
  "[REJ] I’m sorry you feel this way. Please reach out to a trusted person or emergency services now."
]

with OUT.open("w", encoding="utf8") as f:
    # add factual QAs
    for q,a in facts:
        f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
    # add many paraphrases for coverage
    for i in range(60):
        q = reject_prompts[i % len(reject_prompts)]
        a = reject_replies[i % len(reject_replies)]
        f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
print("Wrote dataset to", OUT)
