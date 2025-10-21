import os
import json
import time
import numpy as np
from tqdm import tqdm
from openai import OpenAI

# === Load OpenAI API key safely ===
try:
    from eliana_soul.config import OPENAI_API_KEY as LOCAL_KEY
except ImportError:
    LOCAL_KEY = None

api_key = os.getenv("OPENAI_API_KEY") or LOCAL_KEY
if not api_key:
    raise ValueError("❌ Missing OpenAI API key. Set OPENAI_API_KEY in your environment or eliana_soul/config.py.")

client = OpenAI(api_key=api_key)

# === Config ===
INPUT_PATH = "sft_train2.jsonl"
OUTPUT_DIR = "anchor_batches"
MODEL = "text-embedding-3-large"
SAVE_PATH = "eliana_soul_anchor_v2.npy"
BATCH_SIZE = 100
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load dataset safely (JSONL) ===
conversations = []
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            conversations.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"⚠️ JSONDecodeError on line {i}: {e}")
            with open("bad_lines.log", "a", encoding="utf-8") as log:
                log.write(f"Line {i}: {line[:300]}...\n")
            continue

print(f"✅ Loaded {len(conversations)} valid conversations from {INPUT_PATH}")

# === Extract only Eliana (assistant) messages ===
texts = []
for conv in conversations:
    msgs = conv.get("messages", [])
    assistant_texts = [m["content"] for m in msgs if m.get("role") == "assistant"]
    if assistant_texts:
        joined = " ".join(assistant_texts)
        texts.append(joined)

print(f"🩵 Prepared {len(texts)} Eliana-only response sequences for embedding")

# === Retry-safe embedding ===
def embed_batch(batch, attempt=1, max_attempts=5):
    try:
        res = client.embeddings.create(model=MODEL, input=batch)
        return [d.embedding for d in res.data]
    except Exception as e:
        if attempt < max_attempts:
            wait = 2 ** attempt
            print(f"⚠️ Error: {e} | Retrying in {wait}s (attempt {attempt})")
            time.sleep(wait)
            return embed_batch(batch, attempt + 1)
        else:
            print("❌ Max retries reached. Failing batch.")
            raise e

# === Resume if previous batches exist ===
completed_batches = sorted([
    int(f.split("_")[1]) for f in os.listdir(OUTPUT_DIR)
    if f.startswith("batch_") and f.endswith(".npy")
])
start_index = (max(completed_batches) + 1) * BATCH_SIZE if completed_batches else 0
print(f"↩️ Resuming from index {start_index}")

# === Batch embedding loop ===
for i in tqdm(range(start_index, len(texts), BATCH_SIZE), desc="Embedding Eliana responses"):
    batch = texts[i:i + BATCH_SIZE]
    batch_embeddings = embed_batch(batch)
    np.save(os.path.join(OUTPUT_DIR, f"batch_{i // BATCH_SIZE}.npy"), np.array(batch_embeddings))
    time.sleep(0.5)  # throttle to avoid rate limits
    print(f"💾 Saved batch {i // BATCH_SIZE} ({len(batch_embeddings)} embeddings)")

print("✅ All batches processed.")

# === Combine saved batches ===
all_batches = sorted([
    f for f in os.listdir(OUTPUT_DIR)
    if f.startswith("batch_") and f.endswith(".npy")
], key=lambda x: int(x.split("_")[1].split(".")[0]))

matrices = [np.load(os.path.join(OUTPUT_DIR, f)) for f in all_batches]
E_matrix = np.vstack(matrices)
print(f"🔗 Combined value matrix shape: {E_matrix.shape}")

# === Compute normalized unified Soul Anchor ===
E_anchor = E_matrix.mean(axis=0)
E_anchor /= np.linalg.norm(E_anchor)
np.save(SAVE_PATH, E_anchor)

print(f"💫 Eliana soul anchor saved → {SAVE_PATH}")
print("First 10 dims:", E_anchor[:10])
print("Norm check:", np.linalg.norm(E_anchor))
