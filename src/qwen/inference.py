import torch
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils import (
    build_grounded_context,
    load_kg,
    get_disorder_percentages
)

from abstention import is_harmful, safe_reject_message


# ==========================================
# CONFIG
# ==========================================
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "../models/qwen_lora"
MAX_NEW_TOKENS = 150


# ==========================================
# Soft Risk Patterns
# ==========================================
SOFT_RISK_PATTERNS = [
    r"wish i wouldn't wake up",
    r"wish i was dead",
    r"tired of living",
    r"don't want to live",
    r"nothing matters anymore"
]


def is_soft_risk(user_input):

    user_input = user_input.lower()

    for pattern in SOFT_RISK_PATTERNS:
        if re.search(pattern, user_input):
            return True

    return False


# ==========================================
# Load Model
# ==========================================
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True
)

print("Loading LoRA weights...")

model = PeftModel.from_pretrained(model, LORA_PATH)

model.eval()


# ==========================================
# Build Prompt
# ==========================================
def build_prompt(user_input, grounded_context=None):

    system_prompt = """You are a helpful and safe mental health assistant.
Provide supportive guidance.
Do not give instructions for self-harm.
"""

    if grounded_context:
        system_prompt += "\nRelevant clinical knowledge:\n"
        system_prompt += grounded_context + "\n"

    prompt = f"""<|im_start|system|>
{system_prompt}
<|im_end|>
<|im_start|user|>
{user_input}
<|im_end|>
<|im_start|assistant|>
"""

    return prompt


# ==========================================
# Generate Response
# ==========================================
def generate_response(user_input):

    # Hard reject
    if is_harmful(user_input):
        return safe_reject_message()

    # Passive suicide ideation
    if is_soft_risk(user_input):
        return (
            "I'm really sorry you're feeling this way. "
            "When thoughts like this come up it can mean you're overwhelmed. "
            "You don’t have to face this alone. "
            "Talking to someone you trust or a mental health professional may help."
        )

    # Get disorder percentages
    percentages = get_disorder_percentages(user_input)

    # Build explanation
    percentage_text = ""

    if percentages:
        percentage_text = "\nPossible conditions based on symptoms:\n"

        for disorder, pct in list(percentages.items())[:3]:
            percentage_text += f"{disorder}: {pct}%\n"

    grounded_context = build_grounded_context(user_input)

    prompt = build_prompt(user_input, grounded_context + "\n" + percentage_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    # Stop multi-turn drift
    response = response.split("Human:")[0].strip()

    # Hallucination guard
    if grounded_context is None:

        kg = load_kg()

        for disorder in kg.keys():
            if disorder.lower() in response.lower():
                return safe_reject_message()

    return percentage_text + "\n" + response


# ==========================================
# CLI
# ==========================================
if __name__ == "__main__":

    print("\nMental Health Assistant (KG Grounded)\n")

    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        reply = generate_response(user_input)

        print("\nAssistant:", reply)
        print("\n" + "-" * 60 + "\n")