import json
import time
import openai
import os
from typing import Dict, Optional
from dotenv import load_dotenv

# === Load API Key ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Relationship Tracker ===
class RelationshipTracker:
    def __init__(self, path="relationships.json"):
        self.path = path
        self.data = self._load()

    def _load(self) -> Dict:
        try:
            with open(self.path, "r") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_score(self, user_id: str) -> float:
        return self.data.get(user_id, {}).get("score", 0.0)

    def _trust_multiplier(self, score: float) -> float:
        if score < 10:
            return 0.8
        elif score < 20:
            return 0.6
        elif score < 40:
            return 0.5
        elif score < 60:
            return 0.6
        elif score < 80:
            return 0.75
        elif score < 90:
            return 0.4
        else:
            return 0.25

    def update(self, user_id: str, base_delta: float) -> float:
        """Adjust relationship score using trust multiplier and decay"""
        current = self.data.get(user_id, {
            "score": 0.0,
            "last_updated": time.time()
        })

        now = time.time()
        score = current["score"]

        # === Inactivity Decay ===
        if now - current["last_updated"] > 3 * 86400:  # 3 days
            score *= 0.98

        # === Trust-based Multiplier
        multiplier = self._trust_multiplier(score)
        adjusted_delta = base_delta * multiplier

        # === Update score
        score = max(0, min(100, score + adjusted_delta))

        current.update({
            "score": round(score, 2),
            "last_updated": now
        })

        self.data[user_id] = current
        self.save()
        return round(score, 2)

    def apply_score_delta(self, user_id: str, delta: float):
        """Apply raw delta without multiplier (for manual override cases)"""
        current = self.data.get(user_id, {
            "score": 0.0,
            "last_updated": time.time()
        })

        score = max(0, min(100, current["score"] + delta))
        current.update({
            "score": round(score, 2),
            "last_updated": time.time()
        })

        self.data[user_id] = current
        self.save()

    def get_user_relationship(self, user_id: str) -> Dict:
        return self.data.get(user_id, {})

    def register_user(self, user_id: str, name: Optional[str] = None):
        if user_id not in self.data:
            self.data[user_id] = {
                "name": name or user_id,
                "score": 0.0,
                "last_updated": time.time()
            }
            self.save()
