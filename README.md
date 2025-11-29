# Git Commit Message Generator

AI-powered CLI tool that automatically generates professional git commit messages from code diffs using a fine-tuned Qwen-0.5B language model.

**Live Demo:** [HuggingFace Model](https://huggingface.co/rajtiwariee/auto-commit)

## Features

- **Accurate** - Trained on 100K+ real commit messages
- **Fast** - Generates messages in ~2 seconds
- **Consistent** - Deterministic output (same diff = same message)
- **Clean** - Professional, concise commit messages
- **Easy** - Simple CLI: `commit-gen generate`

## Demo

https://github.com/user-attachments/assets/93114932-3e15-4e68-8a4f-e150caaafa1f

*Click to watch the tool in action - generating commit messages from git diffs*

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/rajtiwariee/GitCommitGenerator
cd GitCommitGenerator

# Install dependencies
pip install -r requirements.txt

# Install CLI tool
pip install -e .
```

### Usage

```bash
# Generate message for staged changes
git add file.py
commit-gen generate

# Generate and auto-commit
commit-gen generate --commit

# Interactive mode (edit before committing)
commit-gen generate --commit --interactive

# View configuration
commit-gen config --show
```

## Model Training

### Dataset Preparation

```bash
# Download CommitPackFT dataset
python scripts/download_dataset.py

# Clean and filter data
python scripts/prepare_dataset.py

# Format for training
python scripts/format_for_training.py
```

### Fine-tuning

```bash
# Train with LoRA (recommended: ~7.5 hours on T4 GPU)
python train.py \
  --num_epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --use_lora \
  --gradient_checkpointing \
  --max_length 384 \
  --use_fp16
```

### Evaluation

```bash
# Run automated metrics
python scripts/test_model.py --num_samples 100

# Interactive testing
python scripts/qualitative_test.py

# Plot training loss
python scripts/plot_loss.py
```

## Model Details

**Base Model:** Qwen-0.5B  
**Training Method:** LoRA (Low-Rank Adaptation)  
**Dataset:** CommitPackFT (55K filtered samples)  
**Hardware:** T4 GPU (16GB VRAM)  
**Training Time:** ~7.5 hours  
**Parameters:** 493M (base) + 35MB (LoRA adapters)

**Training Configuration:**
- Epochs: 3
- Batch Size: 4 (effective: 32 with gradient accumulation)
- Learning Rate: 5e-5
- LoRA r=16, alpha=32
- Mixed Precision (FP32 + FP16)

### Training Loss

![Training Loss](assets/loss_plot.png)

The model converged smoothly from an initial loss of 1.68 to a final loss of ~0.87 over 3 epochs.

## Project Structure

```
GitCommitGenerator/
├── commit_gen/              # CLI tool package
│   ├── cli.py              # Main CLI interface
│   ├── generator.py        # Model inference
│   ├── git_utils.py        # Git integration
│   └── config.py           # Configuration
├── scripts/                # Utilities
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   ├── test_model.py
│   ├── qualitative_test.py
│   └── upload_to_huggingface.py
├── train.py                # Training script
├── setup.py                # Package setup
└── requirements.txt        # Dependencies
```

## Documentation

- **CLI_README.md** - Detailed CLI usage guide
- **DEMO_SCRIPT.md** - Demo/presentation script

## Publishing to HuggingFace

```bash
# Set your token
export HF_TOKEN="your_token_here"

# Upload model
python scripts/upload_to_huggingface.py \
  --model_path ./qwen-0.5b-finetuned/final \
  --repo_name your-username/model-name \
  --metrics_file evaluation_results.json
```

## License

MIT

## Author

Raj Tiwari ([@rajtiwariee](https://github.com/rajtiwariee))
