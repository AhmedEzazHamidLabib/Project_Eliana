from pathlib import Path
import json

# Define new core values to be added
new_core_values = [



]

# If you're not using fragments in this update
new_core_fragments = []

# Path to the core memory file
memory_file_path = Path("core_memory.json")

# Load or initialize the core memory
if memory_file_path.exists():
    with memory_file_path.open("r", encoding="utf-8") as f:
        core_memory = json.load(f)
else:
    core_memory = {"core_fragments": [], "core_values": []}

# Add new core values to the memory
core_memory["core_values"].extend(new_core_values)
core_memory["core_fragments"].extend(new_core_fragments)

# Save the updated memory file
with memory_file_path.open("w", encoding="utf-8") as f:
    json.dump(core_memory, f, ensure_ascii=False, indent=2)

# Output file name for confirmation
memory_file_path.name
