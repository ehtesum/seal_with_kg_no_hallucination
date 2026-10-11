import json
from pathlib import Path

from transformers import GPT2Tokenizer

DATA = Path("eval_predictions.jsonl")

tokenizer = GPT2Tokenizer.from_pretrained("models/seal_gpt2")

latencies = []
token_counts = []

with DATA.open() as f:
    for line in f:
        row = json.loads(line)

        latency = row["latency_ms"]
        prediction = row["prediction"]

        tokens = len(tokenizer.encode(prediction))

        latencies.append(latency)
        token_counts.append(tokens)

avg_latency = sum(latencies) / len(latencies)

total_tokens = sum(token_counts)
total_time_sec = sum(latencies) / 1000

tokens_per_sec = total_tokens / total_time_sec

print("\n=== Performance Metrics ===\n")
print(f"latency_ms: {avg_latency:.2f}")
print(f"tokens_per_sec: {tokens_per_sec:.2f}")