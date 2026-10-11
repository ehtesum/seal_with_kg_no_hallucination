import json
from pathlib import Path

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

DATA = Path("eval_predictions.jsonl")

def normalize(text):
    return text.strip().lower()


def compute_metrics():

    data = [json.loads(x) for x in DATA.open(encoding="utf8")]

    bleu_scores = []
    exact = 0

    rouge1 = []
    rouge2 = []
    rougeL = []

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    smooth = SmoothingFunction().method1

    for row in data:

        ref = normalize(row["reference"])
        pred = normalize(row["prediction"])

        # Exact Match
        if ref == pred:
            exact += 1

        # BLEU
        bleu = sentence_bleu(
            [ref.split()],
            pred.split(),
            smoothing_function=smooth
        )
        bleu_scores.append(bleu)

        # ROUGE
        r = scorer.score(ref, pred)

        rouge1.append(r["rouge1"].fmeasure)
        rouge2.append(r["rouge2"].fmeasure)
        rougeL.append(r["rougeL"].fmeasure)

    n = len(data)

    results = {
        "exact_match": exact / n,
        "bleu": sum(bleu_scores) / n,
        "rouge1": sum(rouge1) / n,
        "rouge2": sum(rouge2) / n,
        "rougeL": sum(rougeL) / n,
    }

    print("\n===== Benchmark Metrics =====\n")

    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    compute_metrics()