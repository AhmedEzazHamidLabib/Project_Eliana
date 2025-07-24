import json

# Load the JSON file with UTF-8 encoding
with open('eliana_major_emotions.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Get all top-level keys
keys = list(data.keys())

# Print them
print(keys)