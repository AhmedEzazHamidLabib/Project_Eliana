from pathlib import Path
import json
import openai
import os

# === Step 1: Load your OpenAI key from environment or config ===
from eliana_soul.config import OPENAI_API_KEY
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY)

# === Step 2: Load input file ===
input_path = Path("eliana_major_emotions.json")
with input_path.open("r", encoding="utf-8") as f:
    emotion_data = json.load(f)

# === Step 3: Prepare data for embedding ===
labels = list(emotion_data.keys())
user_input_checks = [emotion_data[label]["user_input_check"] for label in labels]

# === Step 4: Embed user_input_check fields ===
def embed_texts(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [entry.embedding for entry in response.data]

embeddings = embed_texts(user_input_checks)

# === Step 5: Construct output format ===
embedded_output = []
for i, label in enumerate(labels):
    entry = emotion_data[label]
    embedded_output.append({
        "label": label,
        "embedding": embeddings[i],
        "metadata": {
            "name": label,
            "user_input_check": entry.get("user_input_check", ""),
            "eliana_emotion": entry.get("eliana_emotion", ""),
            "eliana_trait": entry.get("eliana_trait", ""),
            "associated_memory": entry.get("associated_memory", "")
        }
    })

# === Step 6: Save to output file ===
output_path = Path("embedded_eliana_major_emotions.json")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(embedded_output, f, indent=2)

print("✅ embedded_eliana_major_emotions.json generated successfully.")
