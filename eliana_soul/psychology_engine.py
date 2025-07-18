import json
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Any
# === Load key and client ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Cosine similarity ===
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# === Embed input text ===
def embed_input(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding

# === Load embedded patterns from file ===
def load_embedded_patterns(file_path="embedded_psych_models.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# === Get relevant patterns with normalized weights ===
def get_matching_patterns(user_input: str, psych_models: List[Dict], threshold: float = 0.35):
    input_embedding = embed_input(user_input)
    scored = []

    print(f"[🔎] Debug: Checking psychological resonance for input: \"{user_input}\"\n")

    for entry in psych_models:
        score = cosine_similarity(input_embedding, entry["embedding"])
        print(f"[🧠] {entry['label']} → Score: {score:.4f}")  # 👈 This is your debug

        if score >= threshold:
            scored.append({
                "label": entry["label"],
                "score": score,
                "metadata": entry["metadata"]
            })

    if not scored:
        print("[⚠️] No psychological patterns matched above threshold.\n")

    total = sum([s["score"] for s in scored])
    for s in scored:
        s["percent"] = round((s["score"] / total) * 100, 2)

    scored.sort(key=lambda x: x["percent"], reverse=True)
    return scored
