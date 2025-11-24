#!/usr/bin/env python3
"""
Quick quality test - generates messages for hand-picked examples
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Device: {device}\n")

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("./qwen-0.5b-finetuned/final", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "./qwen-0.5b-finetuned/final",
    trust_remote_code=True,
    torch_dtype=torch.float32 if device == "mps" else torch.float16,
).to(device)
model.eval()
print("✅ Model loaded\n")

# Test cases
test_cases = [
    {
        "name": "Bug Fix",
        "diff": """Diff:
File: src/auth.py
Language: Python

Old content:
def login(user):
    if user.password == input_password:
        return True

New content:
def login(user):
    if user and user.password == input_password:
        return True"""
    },
    {
        "name": "New Feature",
        "diff": """Diff:
File: api/routes.py
Language: Python

Old content:
@app.route('/users')
def get_users():
    return jsonify(users)

New content:
@app.route('/users')
def get_users():
    return jsonify(users)

@app.route('/users/<id>')
def get_user(id):
    return jsonify(get_user_by_id(id))"""
    },
    {
        "name": "Refactoring",
        "diff": """Diff:
File: utils.py
Language: Python

Old content:
def process(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result

New content:
def process(items):
    return [item * 2 for item in items if item > 0]"""
    },
    {
        "name": "Documentation",
        "diff": """Diff:
File: README.md
Language: Markdown

Old content:
# Project

Installation:
pip install .

New content:
# Project

## Installation
```bash
pip install -e .
```

## Usage
See docs/ for examples."""
    },
    {
        "name": "Dependency Update",
        "diff": """Diff:
File: requirements.txt
Language: Text

Old content:
numpy==1.20.0
pandas==1.3.0

New content:
numpy==1.24.0
pandas==2.0.0"""
    }
]

print("="*60)
print("QUALITY TEST - Hand-picked Examples")
print("="*60)

for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. {test['name']}")
    print("-" * 40)
    
    prompt = f"Write a git commit message:\n\n{test['diff']}\n\nCommit message:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,  # Even shorter
            do_sample=False,  # Greedy decoding (deterministic)
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract message
    if prompt in generated:
        message = generated[len(prompt):].strip()
    else:
        message = generated.strip()
    
    # Clean up
    import re
    message = message.split('\n')[0].strip()
    message = re.sub(r'\s*\(\d+\).*$', '', message)
    message = re.sub(r'\s*-\s*\d{4}-\d{2}-\d{2}.*$', '', message)
    
    print(f"Generated: {message}")

print("\n" + "="*60)
print("Rate the quality: Are these professional commit messages?")
print("="*60)
