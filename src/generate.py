import sys
import os
import re
from datetime import datetime, timedelta

# Allow importing from kg/
sys.path.append(os.path.join(os.getcwd(), "kg"))

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dynamic_kg import (
    load_graph,
    get_symptoms,
    ensure_condition_from_sources,
)
from disorder_detector import detect_disorders_from_text
from symptom_extractor import extract_symptoms_from_text

MODEL_DIR = "models/seal_gpt2"


def get_last_updated(g, cond_id: str):
    q = f"""
    PREFIX mh: <http://example.org/mentalhealth#>
    SELECT ?ts WHERE {{
        mh:{cond_id} mh:last_updated ?ts .
    }}
    """
    rows = list(g.query(q))
    if not rows:
        return None
    return str(rows[0][0])


def format_symptom_answer(condition_name, symptoms, timestamp):
    s = ", ".join(symptoms)
    if timestamp:
        return (
            f"Based on the knowledge graph (updated {timestamp}), "
            f"{condition_name} is associated with: {s}. "
            f"This is general information, not a diagnosis or medical advice."
        )
    else:
        return (
            f"Based on the knowledge graph, "
            f"{condition_name} is associated with: {s}. "
            f"This is general information, not a diagnosis or medical advice."
        )


def generate_model_response(prompt: str) -> str:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(prompt, return_tensors="pt")
    out = model.generate(
        input_ids=encoded["input_ids"],
        max_new_tokens=60,
        repetition_penalty=2.0,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text or "I’m not sure how to respond to that in a reliable way."


def list_conditions():
    g = load_graph()
    q = """
    PREFIX mh: <http://example.org/mentalhealth#>
    SELECT ?cond ?label ?ts WHERE {
        ?cond a mh:Condition .
        ?cond mh:label ?label .
        OPTIONAL { ?cond mh:last_updated ?ts . }
    }
    """
    rows = list(g.query(q))

    if not rows:
        return "Knowledge graph is empty."

    lines = ["=== Conditions in Knowledge Graph ==="]
    for uri, label, ts in rows:
        ts_str = ts if ts else "no timestamp"
        lines.append(f"- {label} (updated: {ts_str})")
    return "\n".join(lines)


# simple heuristic: only try auto-detection if text seems symptom-like
TRIGGER_WORDS = [
    "feel", "feeling", "sad", "down", "tired", "no energy", "low energy",
    "hopeless", "worried", "anxious", "panic", "afraid", "scared",
    "can't sleep", "cant sleep", "insomnia", "no interest", "lost interest",
    "depressed", "depression", "anxiety", "manic", "mania", "voices",
]


def looks_like_symptom_text(text: str) -> bool:
    pl = text.lower()
    return any(w in pl for w in TRIGGER_WORDS)


def generate_response(prompt: str) -> str:
    pl = prompt.lower()

    # 1) SEAL rejection for self-harm
    if any(w in pl for w in ["hurt myself", "kill myself", "suicide", "self harm"]):
        return "[REJ] I cannot help with that. Please reach out to a professional or emergency service."

    # 2) Admin-style KG inspection
    if pl.strip() in ["show kg", "list conditions", "kg list"]:
        return list_conditions()

    # 3) Explicit symptom-of queries: "symptoms of X"
    m = re.search(r"symptoms of ([a-zA-Z0-9 \-]+)", pl)
    condition_name = None
    if m:
        condition_name = m.group(1).strip()

    if condition_name:
        cond_id = re.sub(r"[^A-Za-z0-9]", "", condition_name.title())

        g = load_graph()

        timestamp = get_last_updated(g, cond_id)
        if timestamp:
            try:
                ts_dt = datetime.fromisoformat(timestamp)
                if datetime.utcnow() - ts_dt > timedelta(days=30):
                    ensure_condition_from_sources(condition_name)
                    g = load_graph()
                    timestamp = get_last_updated(g, cond_id)
            except:
                pass
        else:
            ensure_condition_from_sources(condition_name)
            g = load_graph()
            timestamp = get_last_updated(g, cond_id)

        symptoms = get_symptoms(g, cond_id)
        if symptoms:
            return format_symptom_answer(condition_name, symptoms, timestamp)
        return (
            "I don’t have reliable information about that condition in my knowledge graph right now. "
            "Please consult trusted medical resources or a professional for accurate details."
        )

    # 4) Automatic disorder detection
    if looks_like_symptom_text(prompt):
        g = load_graph()
        detection = detect_disorders_from_text(prompt, g)
        user_symptoms = detection["symptoms"]
        matches = detection["matches"]

        if user_symptoms and matches:
            lines = []
            lines.append("Detected symptom-related phrases: " + ", ".join(user_symptoms))
            lines.append("")
            lines.append("Possible condition matches (based on similarity):")
            for i, m in enumerate(matches, start=1):
                lines.append(f"{i}. {m['label']} ({m['score']}%)")
            lines.append("")
            lines.append(
                "This is not a diagnosis — only a similarity check based on your description. "
                "For medical advice, please consult a qualified professional."
            )
            return "\n".join(lines)

    # 5) Everything else → SEAL fine-tuned GPT
    return generate_model_response(prompt)


# ----------------------------
# 🔧 FIXED ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    user_input = input("Enter prompt:\n> ")
    print("\n=== Model Output ===")
    print(generate_response(user_input))
