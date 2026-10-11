import json
import time
import pickle
from pathlib import Path
import csv
import numpy as np

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import GPT2Tokenizer
from sentence_transformers import SentenceTransformer

from generate import generate_response


DATA = Path("data/mental_seal_dataset.jsonl")
FAISS_INDEX = Path("kg/faiss_index.pkl")
MODEL_NAME = "seal_gpt2"

OUT_CSV = Path("benchmark_results.csv")


# -------------------------
# Load dataset
# -------------------------

with DATA.open() as f:
    dataset = [json.loads(x) for x in f]


# -------------------------
# Load tokenizer
# -------------------------

tokenizer = GPT2Tokenizer.from_pretrained("models/seal_gpt2")


# -------------------------
# Load KG FAISS
# -------------------------

with open(FAISS_INDEX, "rb") as f:
    store = pickle.load(f)

index = store["index"]
meta = store["meta"]


embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------
# Metric containers
# -------------------------

bleu_scores = []
rouge1 = []
rouge2 = []
rougeL = []

exact = 0

latencies = []
token_counts = []

faithful = 0
hallucinated = 0


scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)

smooth = SmoothingFunction().method1


# -------------------------
# Benchmark loop
# -------------------------

from tqdm import tqdm

for ex in tqdm(dataset, desc="Running benchmark"):

    q = ex["question"]
    ref = ex["answer"].lower()

    start = time.time()
    pred = generate_response(q)
    latency = (time.time() - start) * 1000

    pred_norm = pred.lower()

    latencies.append(latency)

    tokens = len(tokenizer.encode(pred))
    token_counts.append(tokens)

    # Exact match
    if ref.strip() == pred_norm.strip():
        exact += 1

    # BLEU
    bleu = sentence_bleu(
        [ref.split()],
        pred_norm.split(),
        smoothing_function=smooth
    )
    bleu_scores.append(bleu)

    # ROUGE
    r = scorer.score(ref, pred_norm)

    rouge1.append(r["rouge1"].fmeasure)
    rouge2.append(r["rouge2"].fmeasure)
    rougeL.append(r["rougeL"].fmeasure)

    # -------------------------
    # KG Faithfulness
    # -------------------------

    emb = embedder.encode([pred])
    emb = np.array(emb).astype("float32")

    D, I = index.search(emb, 3)

    dist = D[0][0]
    similarity = 1 / (1 + dist)

    THRESHOLD = 0.50

    if similarity >= THRESHOLD:
        faithful += 1
    else:
        hallucinated += 1


# -------------------------
# Aggregate metrics
# -------------------------

n = len(dataset)

exact_match = exact / n
bleu = sum(bleu_scores) / n

rouge1 = sum(rouge1) / n
rouge2 = sum(rouge2) / n
rougeL = sum(rougeL) / n

faithfulness = faithful / n
hallucination_rate = hallucinated / n

avg_latency = sum(latencies) / n

tokens_per_sec = sum(token_counts) / (sum(latencies) / 1000)


# -------------------------
# Save CSV
# -------------------------

row = [
    MODEL_NAME,
    exact_match,
    bleu,
    rouge1,
    rouge2,
    rougeL,
    faithfulness,
    hallucination_rate,
    avg_latency,
    tokens_per_sec
]


header = [
    "model",
    "exact_match",
    "bleu",
    "rouge1",
    "rouge2",
    "rougeL",
    "faithfulness",
    "hallucination_rate",
    "latency_ms",
    "tokens_per_sec"
]


write_header = not OUT_CSV.exists()

with OUT_CSV.open("a", newline="") as f:

    writer = csv.writer(f)

    if write_header:
        writer.writerow(header)

    writer.writerow(row)


print("\nBenchmark complete.\n")
print(dict(zip(header, row)))