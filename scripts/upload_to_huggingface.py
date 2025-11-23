#!/usr/bin/env python3
"""
Upload fine-tuned model to Hugging Face Hub
Usage: python scripts/upload_to_huggingface.py --model_path ./qwen-0.5b-finetuned/final --repo_name username/model-name
"""
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def create_model_card(repo_name, metrics=None):
    """Create a model card with metadata"""
    
    metrics_section = ""
    if metrics:
        metrics_section = f"""
## Evaluation Results

- **BLEU Score**: {metrics.get('bleu', 'N/A')}
- **ROUGE-1**: {metrics.get('rouge1', 'N/A')}
- **ROUGE-2**: {metrics.get('rouge2', 'N/A')}
- **ROUGE-L**: {metrics.get('rougeL', 'N/A')}
- **Exact Match Rate**: {metrics.get('exact_match', 'N/A')}
"""
    
    model_card = f"""---
language: en
license: mit
tags:
- code
- git
- commit-message
- qwen2
- lora
base_model: muellerzr/qwen-0.5-git-commit-message-generation
datasets:
- bigcode/commitpackft
---

# Git Commit Message Generator

Fine-tuned [Qwen-0.5B](https://huggingface.co/muellerzr/qwen-0.5-git-commit-message-generation) model for generating professional Git commit messages.

## Model Description

This model was fine-tuned using LoRA (Low-Rank Adaptation) on the CommitPackFT dataset to generate concise, professional commit messages from git diffs.

**Base Model**: `muellerzr/qwen-0.5-git-commit-message-generation`  
**Fine-tuning Method**: LoRA (r=16, alpha=32)  
**Training Data**: 55K filtered commits from CommitPackFT  
**Languages**: Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more

## Intended Use

Generate commit messages for staged changes in a Git repository.

### Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "{repo_name}"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Prepare your diff
diff = \"\"\"
Diff:
File: src/auth.py
Language: Python

Old content:
def login(username, password):
    user = get_user(username)
    if user.password == password:
        return True
    return False

New content:
def login(username, password):
    user = get_user(username)
    if user and user.password == password:
        return True
    return False
\"\"\"

# Generate commit message
prompt = f"Write a git commit message:\\n\\n{{diff}}\\n\\nCommit message:\\n"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )

message = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(message.split("Commit message:")[-1].strip())
# Output: "Fix null check in login function"
```

### CLI Tool

For easier usage, install the companion CLI tool:

```bash
pip install commit-gen
commit-gen generate --commit
```

See the [GitHub repository](https://github.com/YOUR_USERNAME/GitCommitGenerator) for more details.

## Training Details

### Training Data

- **Dataset**: CommitPackFT (filtered subset)
- **Training samples**: 55,730
- **Validation samples**: 6,966
- **Test samples**: 6,967

### Training Procedure

- **Epochs**: 3
- **Batch Size**: 4 (effective batch size: 32 with gradient accumulation)
- **Learning Rate**: 5e-5
- **Optimizer**: AdamW
- **LoRA Config**:
  - r: 16
  - alpha: 32
  - dropout: 0.05
  - target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Hardware

- **GPU**: NVIDIA Tesla T4 (16GB)
- **Precision**: Mixed Precision (FP32 weights + FP16 compute)
- **Training Time**: ~7.5 hours
{metrics_section}

## Limitations

- The model is trained primarily on English commit messages
- Best suited for code changes in common programming languages
- May not handle very large diffs well (>384 tokens)
- Generated messages should be reviewed before committing

## Ethical Considerations

This model is intended to assist developers in writing commit messages, not replace human judgment. Users should:
- Review generated messages for accuracy
- Ensure messages accurately describe the changes
- Follow their team's commit message conventions

## Citation

```bibtex
@misc{{git-commit-generator,
  author = {{Your Name}},
  title = {{Git Commit Message Generator}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{repo_name}}}}},
}}
```

## License

MIT License
"""
    
    return model_card


def upload_model(model_path, repo_name, token, metrics=None, private=False):
    """
    Upload model to Hugging Face Hub
    
    Args:
        model_path: Path to the fine-tuned model
        repo_name: Repository name (username/model-name)
        token: Hugging Face API token
        metrics: Optional dict of evaluation metrics
        private: Whether to make the repo private
    """
    print("=" * 60)
    print("Uploading Model to Hugging Face Hub")
    print("=" * 60)
    
    # Validate paths
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    print(f"\n📦 Model Path: {model_path}")
    print(f"🏷️  Repository: {repo_name}")
    print(f"🔒 Private: {private}\n")
    
    # Initialize API
    api = HfApi()
    
    # Create repository
    print("📝 Creating repository...")
    try:
        create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="model",
        )
        print(f"✅ Repository created: https://huggingface.co/{repo_name}\n")
    except Exception as e:
        print(f"⚠️  Repository may already exist: {e}\n")
    
    # Create and upload model card
    print("📄 Creating model card...")
    model_card_content = create_model_card(repo_name, metrics)
    model_card_path = model_path / "README.md"
    
    with open(model_card_path, 'w') as f:
        f.write(model_card_content)
    print("✅ Model card created\n")
    
    # Upload model files
    print("⬆️  Uploading model files...")
    try:
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            token=token,
            repo_type="model",
            commit_message="Upload fine-tuned model",
        )
        print("✅ Upload complete!\n")
        
        print("=" * 60)
        print(f"🎉 Model successfully uploaded!")
        print(f"🔗 View at: https://huggingface.co/{repo_name}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise


def load_metrics(metrics_file):
    """Load evaluation metrics from JSON file"""
    if not metrics_file or not Path(metrics_file).exists():
        return None
    
    import json
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    return {
        'bleu': f"{data.get('avg_bleu', 0):.4f}",
        'rouge1': f"{data['avg_rouge'].get('rouge1', 0):.4f}",
        'rouge2': f"{data['avg_rouge'].get('rouge2', 0):.4f}",
        'rougeL': f"{data['avg_rouge'].get('rougeL', 0):.4f}",
        'exact_match': f"{data.get('exact_match_rate', 0):.2%}",
    }


def main():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model (e.g., ./qwen-0.5b-finetuned/final)"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Repository name (username/model-name)"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        default="evaluation_results.json",
        help="Path to evaluation metrics JSON"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    
    args = parser.parse_args()
    
    # Get token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Error: No Hugging Face token provided")
        print("   Set HF_TOKEN environment variable or use --token argument")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        return 1
    
    # Load metrics if available
    metrics = load_metrics(args.metrics_file)
    if metrics:
        print(f"📊 Loaded metrics from {args.metrics_file}")
    else:
        print("⚠️  No metrics file found, uploading without evaluation results")
    
    # Upload model
    try:
        upload_model(
            model_path=args.model_path,
            repo_name=args.repo_name,
            token=token,
            metrics=metrics,
            private=args.private,
        )
        return 0
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
