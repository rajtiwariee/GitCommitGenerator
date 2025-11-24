# Git Commit Message Generator

AI-powered tool that generates professional git commit messages from code diffs using a fine-tuned language model.

## Project Structure

```
GitCommitGenerator/
├── docs/                    # Documentation
│   ├── PROJECT_PLAN.md     # Overall project plan
│   ├── FINETUNING_PLAN_0.5B.md  # Detailed fine-tuning guide
│   ├── NEXT_STEPS.md       # Current phase roadmap
│   └── BASELINE_RESULTS.md # Baseline model evaluation
├── scripts/                # Data preparation scripts
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   └── format_for_training.py
├── data/                   # Dataset storage (gitignored)
│   ├── raw/
│   ├── processed/
│   └── formatted/
├── test_base_model.py     # Model testing script
└── requirements.txt       # Dependencies
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
# or with uv
uv add -r requirements.txt
```

### 2. Test Base Model (Optional)

```bash
python test_base_model.py
```

### 3. Prepare Dataset

```bash
# Download CommitPackFT (100K samples, ~10-15 min)
python scripts/download_dataset.py

# Clean and filter data (~5 min)
python scripts/prepare_dataset.py

# Format for training (~2 min)
python scripts/format_for_training.py
```

### 4. Train Model

```bash
# Coming soon: train.py
# Training time: 6-8 hours on T4 GPU
```

## Progress

- [x] Research & Planning
- [x] Environment Setup
- [x] Baseline Testing
- [ ] Dataset Preparation (Current Phase)
- [ ] Model Fine-tuning
- [ ] Evaluation
- [ ] CLI Tool Development

## Model

**Base Model:** `muellerzr/qwen-0.5-git-commit-message-generation` (0.5B parameters)  
**Training Method:** Full fine-tuning  
**Dataset:** CommitPackFT (100K samples)  
**Hardware:** T4 GPU (16GB VRAM)

## Documentation

See `docs/` folder for detailed documentation:
- **PROJECT_PLAN.md** - Overall project plan and research
- **FINETUNING_PLAN_0.5B.md** - Complete fine-tuning guide with code
- **NEXT_STEPS.md** - Current phase instructions
- **BASELINE_RESULTS.md** - Baseline model evaluation

## License

MIT
# Test
