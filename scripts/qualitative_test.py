#!/usr/bin/env python3
"""
Interactive script to test the model on custom diffs.
Usage: python scripts/qualitative_test.py --model_path ./qwen-0.5b-finetuned/final
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_path, device):
    """Load fine-tuned model and tokenizer"""
    print(f"📦 Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}\n")
    return model, tokenizer


def generate_commit_message(model, tokenizer, diff_text, device, temperature=0.7, max_new_tokens=100):
    """Generate a commit message for a given diff"""
    # Create prompt (matching training format)
    prompt = f"Write a git commit message:\n\n{diff_text}\n\nCommit message:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract commit message
    if prompt in generated_text:
        commit_msg = generated_text[len(prompt):].strip()
    else:
        commit_msg = generated_text.strip()
    
    # Take first line as commit message
    commit_msg = commit_msg.split('\n')[0].strip()
    
    return commit_msg


def format_diff(file_path, language, old_content, new_content):
    """Format diff in the training format"""
    diff = f"""Diff:
File: {file_path}
Language: {language}

Old content:
{old_content}

New content:
{new_content}"""
    return diff


def interactive_mode(model, tokenizer, device):
    """Interactive mode for testing"""
    print("="*60)
    print("🤖 Git Commit Message Generator - Interactive Test")
    print("="*60)
    print("\nChoose input mode:")
    print("  1. Paste a git diff directly")
    print("  2. Provide file details (file path, old/new content)")
    print("  3. Use example diffs")
    print("  q. Quit")
    
    while True:
        print("\n" + "-"*60)
        choice = input("\nEnter choice (1/2/3/q): ").strip()
        
        if choice == 'q':
            print("👋 Goodbye!")
            break
        
        elif choice == '1':
            print("\n📝 Paste your git diff (press Ctrl+D or Ctrl+Z when done):")
            print("   Example: git diff HEAD")
            diff_lines = []
            try:
                while True:
                    line = input()
                    diff_lines.append(line)
            except EOFError:
                pass
            
            diff_text = '\n'.join(diff_lines)
            if not diff_text.strip():
                print("⚠️  Empty diff provided.")
                continue
        
        elif choice == '2':
            print("\n📂 Enter file details:")
            file_path = input("  File path (e.g., src/main.py): ").strip()
            language = input("  Language (e.g., Python): ").strip()
            print("  Old content (press Ctrl+D when done):")
            old_lines = []
            try:
                while True:
                    old_lines.append(input())
            except EOFError:
                pass
            print("  New content (press Ctrl+D when done):")
            new_lines = []
            try:
                while True:
                    new_lines.append(input())
            except EOFError:
                pass
            
            diff_text = format_diff(
                file_path,
                language,
                '\n'.join(old_lines),
                '\n'.join(new_lines)
            )
        
        elif choice == '3':
            print("\n📚 Example diffs:")
            print("  1. Bug fix")
            print("  2. New feature")
            print("  3. Refactoring")
            
            example_choice = input("Choose example (1/2/3): ").strip()
            
            if example_choice == '1':
                diff_text = format_diff(
                    "src/auth.py",
                    "Python",
                    "def login(username, password):\n    user = get_user(username)\n    if user.password == password:\n        return True\n    return False",
                    "def login(username, password):\n    user = get_user(username)\n    if user and user.password == password:\n        return True\n    return False"
                )
            elif example_choice == '2':
                diff_text = format_diff(
                    "src/api.py",
                    "Python",
                    "def get_data():\n    return {'status': 'ok'}",
                    "def get_data():\n    return {'status': 'ok'}\n\ndef get_data_by_id(id):\n    return {'id': id, 'status': 'ok'}"
                )
            elif example_choice == '3':
                diff_text = format_diff(
                    "src/utils.py",
                    "Python",
                    "def process(data):\n    result = []\n    for item in data:\n        if item > 0:\n            result.append(item * 2)\n    return result",
                    "def process(data):\n    return [item * 2 for item in data if item > 0]"
                )
            else:
                print("⚠️  Invalid choice.")
                continue
        
        else:
            print("⚠️  Invalid choice.")
            continue
        
        # Generate commit message
        print("\n🤖 Generating commit message...")
        try:
            commit_msg = generate_commit_message(model, tokenizer, diff_text, device)
            print("\n✅ Generated Commit Message:")
            print(f"   {commit_msg}")
        except Exception as e:
            print(f"\n❌ Error generating message: {e}")


def main():
    parser = argparse.ArgumentParser(description="Interactive model testing")
    parser.add_argument("--model_path", type=str, default="./qwen-0.5b-finetuned/final", help="Path to fine-tuned model")
    args = parser.parse_args()
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}\n")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    
    # Start interactive mode
    interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
