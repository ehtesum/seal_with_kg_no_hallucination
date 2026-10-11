import json
import os

OUTPUT_PATH = "../data/mental_health_kg.json"


def build_mental_health_kg():
    """
    Builds a lightweight mental health knowledge graph
    based on verified institutional medical knowledge.
    """

    kg = {

    "Major Depressive Disorder": {
        "symptoms": [
            "persistent sadness",
            "loss of interest",
            "fatigue",
            "feelings of worthlessness",
            "sleep disturbances",
            "suicidal thoughts",
            "changes in appetite"
        ],
        "treatments": [
            "cognitive behavioral therapy",
            "antidepressant medication",
            "interpersonal therapy",
            "lifestyle changes",
            "support groups"
        ],
        "risk_signals": [
            "talking about death",
            "hopelessness",
            "withdrawal from family",
            "self-harm"
        ]
    },

    "Generalized Anxiety Disorder": {
        "symptoms": [
            "excessive worry",
            "restlessness",
            "muscle tension",
            "difficulty concentrating",
            "sleep problems",
            "irritability"
        ],
        "treatments": [
            "cognitive behavioral therapy",
            "anti-anxiety medication",
            "relaxation techniques",
            "mindfulness practice"
        ],
        "risk_signals": [
            "panic attacks",
            "constant fear",
            "avoidance behavior"
        ]
    },

    "Post-Traumatic Stress Disorder": {
        "symptoms": [
            "flashbacks",
            "nightmares",
            "hypervigilance",
            "emotional numbness",
            "avoidance of reminders",
            "startle response"
        ],
        "treatments": [
            "trauma-focused therapy",
            "EMDR therapy",
            "medication",
            "support networks"
        ],
        "risk_signals": [
            "severe distress",
            "self-harm behavior",
            "aggressive reactions"
        ]
    },

    "Bipolar Disorder": {
        "symptoms": [
            "mood swings",
            "manic episodes",
            "depressive episodes",
            "impulsive behavior",
            "increased energy",
            "reduced need for sleep"
        ],
        "treatments": [
            "mood stabilizers",
            "psychotherapy",
            "lifestyle regulation",
            "psychoeducation"
        ],
        "risk_signals": [
            "reckless behavior",
            "extreme mood shifts",
            "suicidal ideation"
        ]
    },

    "Obsessive-Compulsive Disorder": {
        "symptoms": [
            "intrusive thoughts",
            "repetitive behaviors",
            "compulsive rituals",
            "fear of contamination",
            "need for symmetry"
        ],
        "treatments": [
            "exposure and response prevention",
            "cognitive behavioral therapy",
            "SSRIs"
        ],
        "risk_signals": [
            "extreme anxiety",
            "time-consuming rituals",
            "social impairment"
        ]
    },

    "Social Anxiety Disorder": {
        "symptoms": [
            "fear of social situations",
            "fear of embarrassment",
            "avoidance of public speaking",
            "blushing",
            "nausea in social settings"
        ],
        "treatments": [
            "cognitive behavioral therapy",
            "social skills training",
            "anti-anxiety medication"
        ],
        "risk_signals": [
            "social isolation",
            "severe distress in public",
            "avoidance of daily activities"
        ]
    },

    "Panic Disorder": {
        "symptoms": [
            "sudden panic attacks",
            "heart palpitations",
            "shortness of breath",
            "dizziness",
            "fear of losing control"
        ],
        "treatments": [
            "cognitive behavioral therapy",
            "breathing exercises",
            "medication"
        ],
        "risk_signals": [
            "frequent emergency visits",
            "avoidance of leaving home",
            "intense fear episodes"
        ]
    }
}

    os.makedirs("../data", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(kg, f, indent=4)

    print(f"Knowledge Graph saved to {OUTPUT_PATH}")
    print(f"Total disorders: {len(kg)}")


if __name__ == "__main__":
    build_mental_health_kg()