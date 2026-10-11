import json
import pickle
import re
from pathlib import Path

DATA = Path("eval_predictions.jsonl")
INDEX_FILE = Path("kg/faiss_index.pkl")

with open(INDEX_FILE, "rb") as f:
    store = pickle.load(f)

meta = store["meta"]

kg_text = []
for m in meta:
    kg_text.append(m["text"].lower())
    kg_text.append(m["title"].lower())


def split_claims(text):
    return re.split(r"[.,;]", text)


def tokenize(text):
    return re.findall(r"[a-zA-Z]{4,}", text.lower())


supported_claims = 0
unsupported_claims = 0
total_claims = 0


with DATA.open() as f:

    for line in f:

        row = json.loads(line)
        pred = row["prediction"].lower()

        claims = split_claims(pred)

        for claim in claims:

            tokens = tokenize(claim)

            if len(tokens) < 2:
                continue

            total_claims += 1

            supported = False

            for t in tokens:

                for kg in kg_text:

                    if t in kg:
                        supported = True
                        break

                if supported:
                    break

            if supported:
                supported_claims += 1
            else:
                unsupported_claims += 1


faithfulness = supported_claims / total_claims
hallucination_rate = unsupported_claims / total_claims


print("\n=== Fact-Level KG Metrics ===\n")

print("total_claims:", total_claims)
print("supported_claims:", supported_claims)
print("unsupported_claims:", unsupported_claims)

print(f"\nfaithfulness: {faithfulness:.4f}")
print(f"hallucination_rate: {hallucination_rate:.4f}")