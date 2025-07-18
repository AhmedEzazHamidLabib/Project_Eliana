import json
from resonance_engine import (
    build_core_embeddings_with_cache,
    load_core_memory,
    load_core_embeddings
)
import os
from dotenv import load_dotenv

load_dotenv()
print("LOADED:", os.getenv("OPENAI_API_KEY"))

from resonance_engine import (
    build_core_embeddings_with_cache,
    load_core_memory,
    load_core_embeddings
)

core_memory = load_core_memory("core_memory.json")
existing_embeddings = load_core_embeddings("core_embeddings.json")

core_embeddings = build_core_embeddings_with_cache(core_memory, existing_embeddings)

# Save the updated embeddings
with open("core_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(core_embeddings, f, ensure_ascii=False, indent=2)
