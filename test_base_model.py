# test_base_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "muellerzr/qwen-0.5-git-commit-message-generation"

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,  # Updated from torch_dtype
    trust_remote_code=True,  # Allow custom model code
)

# Move to GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print(f"Model loaded! Parameters: {model.num_parameters():,}")
if torch.cuda.is_available():
    print(f"Memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Test inference
test_diff = """diff --git a/utils.py b/utils.py
index 123..456 100644
--- a/utils.py
+++ b/utils.py
@@ -10,3 +10,7 @@ def parse_input(text):
     return processed
+
+def validate_email(email):
+    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
+    return re.match(pattern, email) is not None
"""

# Try simpler, more direct prompt
prompt = f"Write a git commit message:\n{test_diff}"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\nGenerating commit message...")
outputs = model.generate(
    **inputs,
    max_new_tokens=30,  # Shorter to avoid rambling
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,  # Prevent warnings
)

# Decode and extract just the new tokens (skip the prompt)
full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Remove the prompt from the output
message = full_output[len(prompt):].strip()
print(f"\nGenerated commit message:\n{message}")