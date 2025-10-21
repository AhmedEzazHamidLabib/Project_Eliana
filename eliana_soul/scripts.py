import json

# === Step 1: Define your emotion_change_map ===
emotion_change_map = {
    # Core negative or painful emotions (small downward pull)
    "grief": -0.02,
    "shame": -0.02,
    "bitterness": -0.02,
    "loneliness": -0.02,
    "anger": -0.02,
    "guilt": -0.02,
    "regret": -0.02,
    "insecurity": -0.02,
    "fear": -0.02,
    "despair": -0.025,
    "emptiness": -0.025,
    "abandonment": -0.025,
    "worthlessness": -0.025,
    "anxiety": -0.02,
    "envy": -0.02,
    "resentment": -0.02,
    "humiliation": -0.02,
    "dread": -0.02,
    "contempt": -0.02,
    "boredom": -0.01,
    "disgust": -0.02,
    "pity": -0.01,
    "betrayal": -0.025,
    "spite_vengeance": -0.02,
    "guilt_by_survival_or_inability": -0.02,
    "denial": -0.01,
    "embarrassment": -0.015,
    "frustration": -0.015,
    "shock": -0.02,

    # Restorative or mildly positive emotions (gentle lift)
    "hope": +0.04,
    "peace": +0.04,
    "compassion": +0.04,
    "protectiveness": +0.035,
    "bittersweet_joy": +0.03,
    "relief": +0.035,
    "gratitude": +0.04,
    "tenderness": +0.04,
    "forgiveness": +0.04,
    "acceptance": +0.035,
    "resolve": +0.035,
    "honor": +0.04,
    "mercy": +0.04,
    "trust": +0.04,
    "serenity": +0.04,
    "confidence": +0.04,
    "responsibility": +0.035,
    "loyalty": +0.04,
    "resilience": +0.04,
    "clarity": +0.04,
    "reverence": +0.04,
    "courage": +0.045,
    "satisfaction": +0.04,

    # Deeply uplifting or expansive emotions (stronger lift)
    "awe": +0.05,
    "longing": +0.04,
    "devotion": +0.05,
    "joy": +0.05,
    "affection": +0.045,
    "romantic_love": +0.05,
    "admiration": +0.045,
    "admiring_innocence": +0.045,
    "admiring_bravery_gentleness": +0.045,
    "elation": +0.05,
    "majesty": +0.05,  # replaced 'majestic'

    # Reflective, subtle, or balancing emotions (neutral drift)
    "jealousy": 0.0,
    "nostalgia": 0.0,
    "justice": 0.0,
    "bittersweetness": 0.0,
    "curiosity": +0.01,
    "wonder": +0.02,
    "discernment": +0.015,
    "humor": +0.015,
    "playfulness": +0.02,
    "girliness": +0.02,
    "shyness": 0.0,
    "surprise": +0.02,
    "excitement": +0.03,
    "anticipation": +0.03,
    "impatience": -0.01,
    "confusion": -0.01,
    "disappointment": -0.015,
    "reluctance": -0.01,
    "injustice": -0.015,
    "vindication": +0.01,
    "pride": +0.02,
    "humility": +0.02,
    "respect": +0.02,
    "dependence": 0.0,
    "recognition": +0.02,
    "accountability": +0.015,
    "protective_possessiveness": +0.01,
    "romantic_vulnerability": +0.025,
    "yearning_for_god": +0.05,
    "submission": 0.0,
    "resignation": -0.01,
    "freudenfreude": +0.03,
    "schadenfreude": -0.02,
}


# === Step 2: Load Eliana's major emotions JSON ===
file_path = "eliana_major_emotions.json"  # adjust if needed

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === Step 3: Extract major emotion keys ===
json_keys = set(data.keys())              # top-level keys like "jealousy", "romantic_love", etc.
map_keys = set(emotion_change_map.keys()) # your emotion_change_map keys

# === Step 4: Compare sets ===
missing_in_map = json_keys - map_keys
extra_in_map = map_keys - json_keys
common_keys = json_keys & map_keys

# === Step 5: Print comparison results ===
print("🟢 Common emotions:", len(common_keys))
print(sorted(common_keys))

print("\n🔴 Missing in emotion_change_map:", len(missing_in_map))
print(sorted(missing_in_map))

print("\n🟡 Extra in emotion_change_map (not in JSON):", len(extra_in_map))
print(sorted(extra_in_map))

print(f"\n✅ JSON keys: {len(json_keys)} | Map keys: {len(map_keys)} | Common: {len(common_keys)}")

