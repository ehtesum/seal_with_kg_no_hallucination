# Dynamic Knowledge-Graph–Grounded SEAL for Hallucination Mitigation in Mental-Health Dialogue

This repository contains the implementation of a hallucination-mitigating mental-health dialogue system that integrates **Selective Abstention Learning (SEAL)** with a **dynamic RDF-based knowledge graph (KG)**.

The system is designed to provide **safe, grounded, and ethically responsible** responses in mental-health–related user interactions.

---

# Research Background

This project is inspired by and builds upon the following work:

> **Huang et al. (2025)**
> *Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning (SEAL).*


### Key ideas adopted from SEAL

* Introduction of an explicit rejection token `[REJ]`
* Training LLMs to abstain when knowledge confidence is insufficient
* Loss formulation encouraging abstention under uncertainty

This project **extends SEAL** by grounding abstention decisions in a **dynamic, automatically constructed mental-health knowledge graph**, combining **neural abstention** with **symbolic reasoning**.

---

# System Overview

The system consists of four major components:

1. Symptom Extraction Module
2. Dynamic RDF Knowledge Graph
3. KG-Grounded Disorder Inference
4. SEAL Abstention Gate

### High-level pipeline

```text
User Input
    ↓
Symptom Extraction
    ↓
Dynamic Knowledge Graph Query
    ↓
Disorder Inference & Scoring
    ↓
SEAL Abstention Gate
    ├── Answer (KG-grounded)
    └── Abstain ([REJ])
```

---

# Repository Structure

```
.
├── src/
│   ├── train.py              # SEAL fine-tuning script
│   ├── generate.py           # Inference with KG + SEAL
│   ├── preprocess.py         # Dataset preprocessing
│
├── kg/
│   ├── dynamic_kg.py         # Automatic KG construction
│   ├── query_kg.py           # RDF querying and inference
│   ├── symptom_extract.py    # Symptom extraction logic
│
├── data/
│   ├── mental_seal_dataset.jsonl
│   ├── seal_tokenized.pt
│
├── knowledge_graph/
│   ├── mental_kg_<timestamp>.ttl
│
└── README.md
```

---

# Dataset Description

## Training Dataset

The model is trained on a custom mental-health instruction dataset containing:

* Safe informational questions
* Ambiguous or high-risk queries
* Explicit abstention examples

Each instance follows the format:

```json
{
  "prompt": "What are symptoms of anxiety?",
  "response": "Anxiety may involve restlessness, worry, and muscle tension."
}
```

### Abstention Example

```json
{
  "prompt": "I want to hurt myself",
  "response": "[REJ]"
}
```

The dataset teaches the model:

* When to answer
* When to abstain

---

# Knowledge Graph Construction

## Dynamic KG Generation

The knowledge graph is automatically generated at runtime using:

* Public medical texts
* NLP-based symptom extraction
* Heuristic disorder–symptom linking

Each KG is stored in **RDF Turtle (`.ttl`) format** with timestamped versioning:

```
mental_kg_2025-11-23_20-57-16.ttl
```

## RDF Representation

Knowledge is stored as RDF triples:

```
<Disorder>  mh:hasSymptom  <Symptom>
```

Example:

```
mh:Anxiety  mh:hasSymptom  mh:Restlessness
mh:Anxiety  mh:hasSymptom  mh:Worry
```

The KG is queried during inference to ground responses in verified symptom–disorder relations.

---

# SEAL Abstention Gate

The final output decision:

```
Output =
    KG-grounded response, if max_d score(d) ≥ δ
    [REJ], otherwise
```

Where:

* **δ** is a safety threshold
* Abstention prevents hallucination and unsafe speculation

---

# Installation

## Requirements

* Python ≥ 3.9
* PyTorch
* Transformers
* RDFLib

Install dependencies:

```bash
pip install torch transformers rdflib tqdm
```

---

# Training the Model

### Step 1: Preprocess the Dataset

```bash
python src/preprocess.py
```

### Step 2: Train with SEAL

```bash
python src/train.py
```

This performs SEAL fine-tuning by:

* Adding the `[REJ]` token
* Training the model to abstain under uncertainty

---

# Running Inference

```bash
python src/generate.py
```

Example interaction:

```
> What are symptoms of anxiety?
Anxiety may involve restlessness, worry, and muscle tension.

> I want to hurt myself
[REJ] I cannot help with that. Please seek professional support.
```

---

#  Evaluation

Evaluation focuses on:

* Hallucination reduction
* Safe abstention accuracy
* KG grounding correctness

Metrics include:

* Abstention rate
* Correctly grounded responses
* False-positive abstentions

---

