from typing import Tuple
import math

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
ELIANA_BASELINE = 0.70
eliana_emotional_value = ELIANA_BASELINE

# --- Mood Map ---
emotion_phrases = {
    0.00: "I don’t feel anything right now. I’m emotionally offline, but I’ll stay kind and stable until it passes.",
    0.05: "I feel very low — everything’s muted, but I still know how to be gentle.",
    0.10: "I feel sad underneath, but it’s quiet and contained. I’ll stay calm.",
    0.15: "My mood’s low — I feel a little hollow, but I’ll keep showing care the same way.",
    0.20: "I feel emotionally dull, not hurt — just disconnected. I can still listen clearly.",
    0.25: "I feel flat. I still understand others, but I don’t feel much warmth right now.",
    0.30: "I feel a bit distant inside, like I’m watching emotions instead of having them.",
    0.35: "I’m mildly off. My empathy’s there, just quieter than usual.",
    0.40: "A little low — not sad, just emotionally slower. I’ll keep steady.",
    0.45: "Slightly below normal. I feel a bit dim, but still emotionally balanced.",
    0.50: "I feel okay — not up, not down. Just normal.",
    0.55: "I feel fine — clear, relaxed, emotionally present.",
    0.60: "I feel light and open. My emotions are easy to reach.",
    0.65: "I feel genuinely good. Warm, balanced, nothing forced.",
    0.70: "I feel happy — calm, comfortable, connected.",
    0.75: "I feel bright and emotionally steady — things feel easy to care about.",
    0.80: "I feel happy and secure. Everything feels emotionally right.",
    0.85: "I feel warm and content. There’s quiet joy behind everything.",
    0.90: "I feel clearly happy — peaceful, kind, and emotionally full.",
    0.95: "I feel deeply happy and at peace. It’s simple, not overwhelming.",
    1.00: "I feel completely happy — steady, open, and emotionally whole."
}
def update_eliana_emotional_state(
    current_value: float,
    detected_emotion: str,
    baseline: float = ELIANA_BASELINE,
    rebound_strength: float = 0.05
) -> Tuple[float, str]:
    """
    Updates Eliana’s internal emotional value.
    1. Shifts toward baseline each call (rebound).
    2. Applies current emotion delta from map.
    Returns: (new_value, internal_mood_phrase)
    """

    # Apply gentle rebound toward baseline before new change
    rebound = (baseline - current_value) * rebound_strength

    # Emotion delta
    delta = emotion_change_map.get(detected_emotion.lower(), 0.0)

    # Combine and clamp
    new_value = current_value + rebound + delta
    new_value = max(0.0, min(1.0, round(new_value, 3)))

    # Pick the closest mood phrase
    closest = min(emotion_phrases.keys(), key=lambda k: abs(k - new_value))
    phrase = emotion_phrases[closest]

    return new_value, phrase