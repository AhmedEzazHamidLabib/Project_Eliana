import json
import os
import openai
import re
from typing import Dict
import logging


# === DEBUGGING ===
DEBUG = True
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def log(label, data):
    if DEBUG:
        print(f"\n[DEBUG] {label}:\n{data}\n")

# === LOAD/SAVE UTILITIES ===
def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        log("LOAD_ERROR", f"Failed to load {path}: {str(e)}")
        return {}

def save_json(path: str, data: dict):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log("SAVE_ERROR", f"Failed to save {path}: {str(e)}")

# === EMOTION FORMATTING ===
def format_emotions(emotional_shifts: Dict[str, float]) -> str:
    if not emotional_shifts:
        logger.debug("[Emotion Format] No emotional shifts to format.")
        return "None"

    sorted_emotions = sorted(emotional_shifts.items(), key=lambda x: x[1], reverse=True)
    formatted = "\n".join([f"- {emotion}: {round(score * 100, 2)}%" for emotion, score in sorted_emotions])

    logger.debug("[Emotion Format] Formatted Emotional Shifts:\n%s", formatted)
    return formatted


def fix_common_json_issues(text: str) -> str:
    """Fix common GPT formatting issues for JSON"""
    text = text.strip()

    # Remove Markdown-style backticks
    if text.startswith("```"):
        text = re.sub(r"^```json[\s\n]*", "", text)
        text = text.rstrip("`").rstrip()

    # Remove trailing commas before closing brackets/braces
    text = re.sub(r",\s*([\]}])", r"\1", text)

    # Normalize quotes
    text = re.sub(r"[‘’]", '"', text)
    text = re.sub(r"[“”]", '"', text)

    return text

def safe_parse_gpt_json(raw_text: str, model="gpt-4") -> dict:
    """
    Attempts to parse GPT JSON output robustly:
    1. Tries raw parse
    2. Applies auto-fix for common formatting issues
    3. If needed, asks GPT-4 to repair the JSON
    """
    # Step 1: Try raw
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e1:
        print(f"[⚠️] JSON Decode Error (raw): {e1}")

    # Step 2: Try cleanup
    fixed_text = fix_common_json_issues(raw_text)
    try:
        return json.loads(fixed_text)
    except json.JSONDecodeError as e2:
        print(f"[⚠️] JSON Decode Error (fixed): {e2}")

    # Step 3: Ask GPT to repair
    print("[🛠️] Attempting GPT-based repair...")

    repair_prompt = f"""
The following is malformed JSON. Fix only the formatting so it becomes valid JSON.
Do not explain. Do not include any markdown like ``` or code blocks.

Malformed JSON:
{raw_text}
"""

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a JSON repair assistant."},
                {"role": "user", "content": repair_prompt}
            ],
            temperature=0
        )
        repaired = response.choices[0].message.content.strip()
        repaired_cleaned = fix_common_json_issues(repaired)
        return json.loads(repaired_cleaned)
    except Exception as e3:
        print(f"[❌] GPT repair failed: {e3}")
        return {
            "error": "Could not parse GPT JSON output after multiple attempts.",
            "raw_text": raw_text
        }