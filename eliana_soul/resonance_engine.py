from typing import List, Dict, Optional, Tuple
import openai
import numpy as np
import json
import os
from pathlib import Path
from eliana_soul.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY
EMBEDDING_MODEL = "text-embedding-ada-002"

def embed_text(text: str) -> List[float]:
    response = openai.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_core_memory(filename: str) -> Dict:
    filepath = Path(__file__).resolve().parent / filename
    with filepath.open('r', encoding='utf-8') as f:
        return json.load(f)

def load_core_embeddings(filename: str) -> List[Dict]:
    filepath = Path(__file__).resolve().parent / filename
    if filepath.exists():
        with filepath.open('r', encoding='utf-8') as f:
            return json.load(f)
    return []

def build_core_embeddings(core_memory: Dict) -> List[Dict]:
    core_embeddings = []

    for value in core_memory.get("core_values", []):
        embedding = embed_text(value["anchor"])
        core_embeddings.append({
            "type": "value",
            "text": value["anchor"],
            "metadata": value,
            "embedding": embedding
        })

    for frag in core_memory.get("core_fragments", []):
        embedding = embed_text(frag["summary"])
        core_embeddings.append({
            "type": "fragment",
            "text": frag["summary"],
            "metadata": frag,
            "embedding": embedding
        })

    return core_embeddings

def build_core_embeddings_with_cache(core_memory: Dict, existing_embeddings: Optional[List[Dict]] = None) -> List[Dict]:
    if existing_embeddings is None:
        existing_embeddings = []

    prev_map: Dict[Tuple[str, str], Dict] = {
        (item["type"], item["text"]): item
        for item in existing_embeddings
    }

    core_embeddings = []

    for value in core_memory.get("core_values", []):
        text = value["anchor"]
        key = ("value", text)
        if key in prev_map:
            core_embeddings.append(prev_map[key])
        else:
            embedding = embed_text(text)
            core_embeddings.append({
                "type": "value",
                "text": text,
                "metadata": value,
                "embedding": embedding
            })

    for frag in core_memory.get("core_fragments", []):
        text = frag["summary"]
        key = ("fragment", text)
        if key in prev_map:
            core_embeddings.append(prev_map[key])
        else:
            embedding = embed_text(text)
            core_embeddings.append({
                "type": "fragment",
                "text": text,
                "metadata": frag,
                "embedding": embedding
            })

    return core_embeddings


def get_top_resonances(
    user_text: str,
    core_embeddings: List[Dict],
    top_values_k: int = 1,
    top_frags_k: int = 1,
    value_threshold: float = 0.4,
    fragment_threshold: float = 0.5
) -> List[Dict]:
    user_embedding = embed_text(user_text)
    value_results = []
    fragment_results = []

    for item in core_embeddings:
        sim = cosine_similarity(user_embedding, item["embedding"])

        if item["type"] == "value" and sim >= value_threshold:
            value_results.append({
                "type": "value",
                "text": item["text"],
                "score": sim,
                "metadata": item["metadata"]
            })
        elif item["type"] == "fragment" and sim >= fragment_threshold:
            fragment_results.append({
                "type": "fragment",
                "text": item["text"],
                "score": sim,
                "metadata": item["metadata"]
            })

    value_results = sorted(value_results, key=lambda x: x["score"], reverse=True)[:top_values_k]
    fragment_results = sorted(fragment_results, key=lambda x: x["score"], reverse=True)[:top_frags_k]

    return value_results + fragment_results
