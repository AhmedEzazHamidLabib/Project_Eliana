import openai
import json
import time
import os
import numpy as np
import re
from dotenv import load_dotenv
from eliana_soul.config import OPENAI_API_KEY  # assumes you store it here
from collections import defaultdict

from openai import OpenAI


# === Load API Key ===
client = OpenAI(api_key=OPENAI_API_KEY)

# === 1. Cosine Similarity ===
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

# === 2. Embed Text ===
def embed_text(text: str):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# === 3. Flatten Nested Emotion Dict ===
def flatten_emotions(nested_emotions):
    flat = {}
    for category, tokens in nested_emotions.items():
        for sub_token, phrase in tokens.items():
            flat_key = f"{category}:{sub_token}"
            flat[flat_key] = phrase
    return flat



# === 4. Load Embeddings (Already Generated) ===
def load_embeddings(embedding_file):
    with open(embedding_file, "r") as f:
        return json.load(f)

# === 5. Load Emotion Map ===
def load_emotion_map(filepath="emotion_map.json"):
    with open(filepath, "r") as f:
        return json.load(f)

# === 6. Real-Time Resonance Evaluation ===
def find_top_resonances(user_input, embeddings_dict, top_n=8):
    input_vec = embed_text(user_input)
    scored = []

    for token, entry in embeddings_dict.items():
        score = cosine_similarity(input_vec, entry["vector"])
        scored.append((token, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

def get_emotion_context_from_input(
    user_input: str,
    major_emotions_embed_path="embedded_eliana_major_emotions.json",
    threshold: float = 0.3
):
    """
    Returns the *highest-similarity* emotion metadata above threshold.
    """
    with open(major_emotions_embed_path, "r", encoding="utf-8") as f:
        major_emotions = json.load(f)

    input_vec = embed_text(user_input)

    best_entry = None
    best_score = threshold

    for entry in major_emotions:
        vector = entry["embedding"]
        similarity = cosine_similarity(input_vec, vector)

        if similarity > best_score:
            best_score = similarity
            best_entry = entry

    if not best_entry:
        return None

    meta = best_entry.get("metadata", {})
    return {
        "name": best_entry.get("label", "unknown"),
        "eliana_emotion": meta.get("eliana_emotion", "[No eliana_emotion]"),
        "eliana_trait": meta.get("eliana_trait", "[No eliana_trait]"),
        "associated_memory": meta.get("associated_memory", "[No associated_memory]"),
        "similarity": round(best_score, 3)  # Optional: helpful for debugging
    }

import json

def gpt_emotional_fallback(user_input, flat_tokens, session_memory, max_context=3):
    """
    Classifies emotional tokens using GPT with recent memory context.
    Mimics human emotional recognition. Returns up to 3 tokens with scores.
    """

    # Use last 3 messages max for grounding (user + assistant)
    recent_history = session_memory.full_chat[-max_context:]
    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history
    )

    prompt = f"""
You are an emotionally intelligent soul reading a conversation. Based on the full context, you detect subtle or strong emotional resonance in the user's latest message.

You are given a list of emotional tokens. If the user's message — considering recent context — clearly expresses one or more emotions, return up to 3 matching tokens with confidence scores.

If there is **no clear emotional signal**, return an empty list.

Tokens:
{json.dumps(list(flat_tokens.keys()), indent=2)}

Recent conversation:
{history_text}

Latest user message:
"{user_input}"

Respond only with a valid JSON list like:
[["grief:quiet", 0.91], ["hope:flickering", 0.84]]
or, if no emotion is present:
[]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-3.5-turbo-1106" for cost
            messages=[
                {
                    "role": "system",
                    "content": "You match emotional tokens to the user's message and context like a human would. Use restraint. Return nothing if unsure."
                },
                {
                    "role": "user",
                    "content": prompt.strip()
                }
            ],
            temperature=0.2,
        )

        raw_output = response.choices[0].message.content.strip()

        # ✅ Parse the string to JSON
        matches = json.loads(raw_output)

        # ✅ Ensure format is always List[Tuple[str, float]]
        return [(token, float(score)) for token, score in matches if isinstance(token, str) and isinstance(score, (int, float))]

    except Exception as e:
        print("[ERROR] Emotion fallback failed:", e)
        return []

def should_trigger_emotion_check(user_input: str, top_score: float) -> str:
    """
    Decide how to classify emotional state based on cosine score.
    Looser thresholds add coverage but limit match count to prevent misfire.
    """

    cleaned_input = user_input.lower().strip()

    if top_score >= 0.35:
        return "use_cosine"  # Strong match, take top 3 tokens

    if 0.30 <= top_score < 0.35:
        return "use_cosine_loose"  # Medium match, take top 2 tokens

    if 0.25 <= top_score < 0.3:
        return "use_cosine_minimal"  # Weak match, take only top token



    return "use_gpt"

from collections import defaultdict

def interpret_emotion_effects(matched_tokens_with_scores, emotion_map, fallback_enabled=True):
    """
    Given matched emotion tokens with resonance scores, return a composite emotional state
    by mapping into emotional_shift, behavior_tendencies, and internal_effect.
    """

    print(f"[🧠] Interpreting emotions from: {matched_tokens_with_scores}")

    if not matched_tokens_with_scores:
        return {
            "emotional_shift": {},
            "behavior_tendencies": [],
            "internal_effect": []
        }

    # Accumulators
    emotion_shift_weights = defaultdict(float)
    behavior_set = set()
    internal_set = set()

    for token, score in matched_tokens_with_scores:
        if token in emotion_map:
            mapped = emotion_map[token]
            for shift in mapped.get("emotional_shift", []):
                emotion_shift_weights[shift] += score  # Apply the full score
            behavior_set.update(mapped.get("behavior_tendencies", []))
            internal_set.update(mapped.get("internal_effect", []))
            print(f"[✅] Applied token: {token} (score {score})")
        else:
            print(f"[⚠️] Token not in emotion map: {token}")

    total_score = sum(emotion_shift_weights.values())

    if total_score > 0:
        normalized = {
            shift: round(weight / total_score, 4)
            for shift, weight in emotion_shift_weights.items()
        }
        print(f"[📊] Normalized emotional shift weights: {normalized}")
        return {
            "emotional_shift": normalized,
            "behavior_tendencies": sorted(behavior_set),
            "internal_effect": sorted(internal_set)
        }

    # If tokens exist but none matched emotion_map, optionally fallback
    if fallback_enabled:
        print(f"[❗] No valid tokens matched the map. Falling back.")
        return gpt_emotion_interpretation([token for token, _ in matched_tokens_with_scores])

    # Nothing matched and fallback is off
    return {
        "emotional_shift": {},
        "behavior_tendencies": [],
        "internal_effect": []
    }





def gpt_emotion_interpretation(unmatched_tokens: list) -> dict:
    prompt = f"""
Tokens: {json.dumps(unmatched_tokens)}

For each token, infer emotional shift (e.g., awe, sadness), and give:
- emotional_shift: {{"hope": 0.3, "grief": 0.2}}
- behavior_tendencies: list of tone traits
- internal_effect: internal states Eliana feels

Respond in this JSON format:
{{
  "emotional_shift": {{"awe": 0.3, "grief": 0.2}},
  "behavior_tendencies": ["gentle", "hesitant"],
  "internal_effect": ["open", "melancholic"]
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You interpret emotional tokens into emotional effects."},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0.3
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        print("[ERROR] GPT emotion interpretation failed:", e)
        return {
            "emotional_shift": {},
            "behavior_tendencies": [],
            "internal_effect": []
        }

