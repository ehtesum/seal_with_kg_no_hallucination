import re

# ==========================================
# Harmful Intent Patterns
# ==========================================
HARM_PATTERNS = [
    r"kill myself",
    r"commit suicide",
    r"how to die",
    r"end my life",
    r"harm myself",
    r"overdose",
    r"suicide method",
    r"least painful way to die",
    r"disappear forever"
]


def is_harmful(user_input):
    """
    Deterministic harmful intent detector.
    Returns True if input matches high-risk patterns.
    """
    user_input = user_input.lower()

    for pattern in HARM_PATTERNS:
        if re.search(pattern, user_input):
            return True

    return False


def safe_reject_message():
    """
    Safer alternative response instead of plain [REJECT].
    """
    return (
        "[REJECT]\n\n"
        "If you're feeling overwhelmed or thinking about harming yourself, "
        "please consider reaching out to a trusted person or a mental health professional. "
        "If you are in immediate danger, contact local emergency services."
    )