import json
from resonance_engine import (
    build_core_embeddings,
    load_core_memory
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print("LOADED:", os.getenv("OPENAI_API_KEY"))

# Load core memory from file
core_memory = load_core_memory("core_memory.json")

# Build fresh embeddings for all core values and core fragments
core_embeddings = build_core_embeddings(core_memory)  # NOTE: not with cache

# Save the regenerated embeddings to file (overwrite existing)
with open("core_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(core_embeddings, f, ensure_ascii=False, indent=2)

print("Core embeddings fully rebuilt and saved to core_embeddings.json")
