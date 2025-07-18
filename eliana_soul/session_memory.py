from typing import List, Dict, Optional
import time
from datetime import timezone, datetime

class SessionMemory:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt


        self.important_facts: List[Dict[str, str]] = []

        # Full emotional memory
        self.user_emotions: List[Dict[str, float]] = []  # {"emotion": "grief", "score": 0.8}
        self.eliana_emotions: List[Dict[str, float]] = []

        # Recent emotions (last 3)
        self.recent_user_emotions: List[Dict[str, float]] = []
        self.recent_eliana_emotions: List[Dict[str, float]] = []

        self.last_user_message: Optional[str] = ""
        self.last_user_messages: List[str] = []  # stores last 2 user messages
        self.last_eliana_reply: Optional[str] = ""

        self.psychological_patterns: List[Dict] = []




        # Resonance memory
        self.core_value_resonance: List[Dict[str, float]] = []
        self.core_fragment_resonance: List[Dict[str, float]] = []

        # Dialogue memory
        self.recent_messages: List[Dict[str, str]] = []
        self.full_chat: List[Dict[str, str]] = []

        self.session_start_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.session_end_time: Optional[str] = None

        self.personality_trace: List[Dict] = []



    def store_important_fact(self, text: str):
        self.important_facts.append({"fact": text, "timestamp": time.time()})

    def store_emotion(self, emotion: str, score: float = 0.7, source: str = "user"):
        entry = {"emotion": emotion, "score": score, "timestamp": time.time()}
        if source == "user":
            self.user_emotions.append(entry)
            self.recent_user_emotions.append(entry)
            self.recent_user_emotions = self.recent_user_emotions[-3:]
            self._update_mood()
        elif source == "eliana":
            self.eliana_emotions.append(entry)
            self.recent_eliana_emotions.append(entry)
            self.recent_eliana_emotions = self.recent_eliana_emotions[-3:]


    def add_user_message(self, content: str):
        self.full_chat.append({"role": "user", "content": content})
        self.recent_messages.append({"role": "user", "content": content})
        self.last_user_message = content
        self._trim_recent()

    def add_assistant_message(self, content: str):
        self.full_chat.append({"role": "assistant", "content": content})
        self.recent_messages.append({"role": "assistant", "content": content})
        self._trim_recent()

    def _trim_recent(self, max_len: int = 6):
        if len(self.recent_messages) > max_len:
            self.recent_messages = self.recent_messages[-max_len:]

    def store_core_value_resonance(self, anchor: str, score: float, threshold: float = 0.9):
        entry = {"anchor": anchor, "score": score}
        self.core_value_resonance.append(entry)
        if score >= threshold:
            self.store_important_fact(f"Core value resonated: {anchor} ({score:.2f})")

    def store_core_fragment_resonance(self, anchor: str, score: float, threshold: float = 0.9):
        entry = {"anchor": anchor, "score": score}
        self.core_fragment_resonance.append(entry)
        if score >= threshold:
            self.store_important_fact(f"Core fragment resonated: {anchor} ({score:.2f})")





    def _update_mood(self):
        if self.recent_user_emotions:
            sorted_moods = sorted(self.recent_user_emotions, key=lambda x: x["score"], reverse=True)
            top = sorted_moods[0]
            self.user_mood_state = {"dominant_emotion": top["emotion"], "intensity": top["score"]}

    def build_summary(self) -> str:
        lines = []

        if self.important_facts:
            lines.append("Important facts: " + " | ".join(f["fact"] for f in self.important_facts))

        if self.recent_user_emotions:
            emo_str = ", ".join(
                f"{e['emotion']} ({round(e['score'], 2)})" if isinstance(e, dict)
                else f"{e[0]} ({round(e[1], 2)})"
                for e in self.recent_user_emotions
            )
            lines.append("Recent user emotions: " + emo_str + ".")

        if self.recent_eliana_emotions:
            for i, emo in enumerate(self.recent_eliana_emotions, 1):
                shifts = emo.get("emotional_shift", {})
                shift_str = ", ".join(
                    f"{emotion} ({round(score * 100, 1)}%)" for emotion, score in shifts.items()
                )
                lines.append(f"Eliana Emotion {i}: {shift_str}.")

        return "\n".join(lines)

    def build_prompt(self) -> List[Dict[str, str]]:
        prompt = [{"role": "system", "content": self.system_prompt}]
        prompt += self.recent_messages[-6:]  # Most recent 3 pairs (user + Eliana)
        return prompt

    def get_personality_trace(self) -> List[Dict]:
        return self.personality_trace

    def update_recent_history(self):
        self.last_user_messages = [
                                      msg["content"] for msg in self.full_chat if msg["role"] == "user"
                                  ][-2:]

        self.last_eliana_reply = next(
            (msg["content"] for msg in reversed(self.full_chat)
             if msg["role"] == "assistant"),
            ""
        )



