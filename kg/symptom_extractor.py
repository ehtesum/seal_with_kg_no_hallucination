import re
from typing import List, Dict

# Canonical symptom phrases to detect in user text.
# These should match or be very close to what dynamic_kg stores.
SYM_PHRASES: Dict[str, list[str]] = {
    "sadness": ["sad", "sadness", "feeling down", "feeling low"],
    "low mood": ["low mood", "down mood"],
    "hopelessness": ["hopeless", "no hope", "hopelessness"],
    "elevated mood": ["elevated mood", "too happy", "overly happy"],
    "irritability": ["irritable", "irritability", "easily annoyed", "short-tempered"],
    "mood swings": ["mood swings", "up and down", "emotional ups and downs"],
    "mania": ["manic", "mania"],
    "hypomania": ["hypomanic", "hypomania"],

    "racing thoughts": ["racing thoughts", "thoughts racing", "mind is racing"],
    "slowed thinking": ["slowed thinking", "thinking is slow"],
    "poor concentration": ["can't concentrate", "difficulty concentrating", "poor concentration"],

    "delusions": ["delusions", "delusional"],
    "hallucinations": ["hallucinations", "hearing voices", "seeing things that are not there"],
    "disorganized thinking": ["disorganized thinking", "thoughts feel jumbled"],

    "fatigue": ["fatigue", "tired all the time", "always tired", "exhausted"],
    "low energy": ["no energy", "low energy", "lack of energy"],
    "excessive energy": ["too much energy", "hyper", "overly energetic"],
    "restlessness": ["restless", "restlessness", "can't sit still"],

    "sleep problems": ["sleep problems", "trouble sleeping", "sleep issues"],
    "insomnia": ["insomnia", "can't sleep"],
    "reduced need for sleep": ["reduced need for sleep", "sleeping less but not tired"],
    "oversleeping": ["sleeping too much", "oversleeping"],

    "worry": ["worried", "worrying", "worry"],
    "fear": ["afraid", "fearful", "fear"],
    "panic": ["panic", "panic attacks", "panicking"],
    "muscle tension": ["muscle tension", "tense muscles"],

    "loss of interest": ["no interest in things", "lost interest", "loss of interest"],
    "loss of pleasure": ["no pleasure", "can't enjoy things", "loss of pleasure"],

    "withdrawal": ["pulling away", "withdrawing", "keeping away from people"],
    "social withdrawal": ["avoiding people", "social withdrawal", "isolating"],
    "agitation": ["agitated", "agitation", "on edge"],
    "risky behavior": ["risky behavior", "doing risky things"],
    "impulsivity": ["impulsive", "impulsivity"],
    "repetitive behaviors": ["repeating actions", "rituals", "repetitive behaviors"],

    "changes in appetite": ["changes in appetite", "eating more or less"],
    "increased appetite": ["increased appetite", "eating more than usual"],
    "reduced appetite": ["reduced appetite", "eating less"],

    "winter depression": ["winter depression", "feel worse in winter", "seasonal depression"],
    "light sensitivity": ["sensitive to light", "light makes me feel worse"],

    "difficulty concentrating": ["difficulty concentrating", "hard to focus"],
    "difficulty functioning": ["hard to function", "struggling to function"],
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_symptoms_from_text(text: str) -> List[str]:
    """
    Extracts symptom labels based on simple keyword and phrase rules.
    Returns canonical symptom names that can be matched against KG.
    """
    text = normalize_text(text)
    found = set()

    for canonical, phrases in SYM_PHRASES.items():
        for phrase in phrases:
            if phrase in text:
                found.add(canonical)
                break

    return sorted(found)