import openai
import os
import json

from eliana_soul.config import OPENAI_API_KEY

# === Load your OpenAI key from environment ===

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")or OPENAI_API_KEY)

# === Step 1: Load input file ===
with open("psychological_models.json", "r", encoding="utf-8") as f:
    models = json.load(f)

# === Step 2: Prepare data ===
labels = list(models.keys())
descriptions = [models[label]["description"] for label in labels]

# === Step 3: Embed using new SDK ===
def embed_texts(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [entry.embedding for entry in response.data]

embeddings = embed_texts(descriptions)

# === Step 4: Build output ===
embedded_output = []
for i, label in enumerate(labels):
    entry = models[label]
    embedded_output.append({
        "label": label,
        "embedding": embeddings[i],
        "metadata": {
            "description": entry.get("description", ""),
            "root_cause": entry.get("root_cause", ""),
            "therapist_fix": entry.get("therapist_fix", ""),
            "eliana_assistance": entry.get("eliana_assistance", "")
        }
    })

# === Step 5: Save output ===
with open("embedded_psych_models.json", "w", encoding="utf-8") as f:
    json.dump(embedded_output, f, indent=2)

print("✅ embedded_psych_models.json generated successfully.")
