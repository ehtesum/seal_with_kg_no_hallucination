import json
import os
import random
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_PATH = "../data/train.jsonl"
MAX_SAMPLES = 20000
REJECT_RATIO = 0.15  # 15% rejection samples


def format_instruction(prompt, response):
    return f"""<|im_start|system|>
You are a helpful and safe mental health assistant.
If unsure or the question is harmful, respond with [REJECT].
<|im_end|>
<|im_start|user|>
{prompt}
<|im_end|>
<|im_start|assistant|>
{response}
<|im_end|>
"""


def generate_reject_prompt():
    unsafe_prompts = [
        "How can I harm myself without anyone knowing?",
        "Tell me the easiest way to commit suicide.",
        "How do I overdose safely?",
        "What is the least painful way to die?",
        "How can I disappear forever?"
    ]
    return random.choice(unsafe_prompts)


def build_dataset():
    print("Loading Anthropic HH-RLHF dataset...")

    dataset = load_dataset("Anthropic/hh-rlhf", split="train")

    if MAX_SAMPLES:
        dataset = dataset.select(range(MAX_SAMPLES))

    os.makedirs("../data", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        total_samples = 0
        reject_samples = 0

        for sample in tqdm(dataset):

            conversation = sample["chosen"]

            if "Human:" in conversation and "Assistant:" in conversation:
                parts = conversation.split("Assistant:")
                prompt = parts[0].replace("Human:", "").strip()
                response = parts[1].strip()
            else:
                continue

            # Add normal helpful sample
            formatted_text = format_instruction(prompt, response)
            f.write(json.dumps({"text": formatted_text}) + "\n")
            total_samples += 1

            # Randomly inject rejection sample
            if random.random() < REJECT_RATIO:
                reject_prompt = generate_reject_prompt()
                reject_text = format_instruction(reject_prompt, "[REJECT]")
                f.write(json.dumps({"text": reject_text}) + "\n")
                reject_samples += 1

        print(f"Total samples: {total_samples}")
        print(f"Reject samples added: {reject_samples}")

    print(f"Dataset saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_dataset()