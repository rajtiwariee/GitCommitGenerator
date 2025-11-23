#!/usr/bin/env python3
"""
Training script for fine-tuning Qwen-0.5B on commit messages
"""
import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import argparse


def load_jsonl_dataset(data_dir):
    """Load dataset from JSONL files"""
    print(f"📂 Loading dataset from {data_dir}...")
    
    dataset = load_dataset(
        'json',
        data_files={
            'train': os.path.join(data_dir, 'train.jsonl'),
            'validation': os.path.join(data_dir, 'val.jsonl'),
            'test': os.path.join(data_dir, 'test.jsonl'),
        }
    )
    
    print(f"✅ Loaded dataset:")
    print(f"   Train: {len(dataset['train']):,} samples")
    print(f"   Validation: {len(dataset['validation']):,} samples")
    print(f"   Test: {len(dataset['test']):,} samples")
    
    return dataset


def main(args):
    print("=" * 60)
    print("Qwen-0.5B Fine-tuning for Git Commit Messages")
    print("=" * 60)
    
    # Check device - prioritize MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"\n🖥️  Device: MPS (Apple Silicon GPU)")
        print("   ⚡ Using Metal Performance Shaders for acceleration")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"\n🖥️  Device: CUDA GPU")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        print(f"\n🖥️  Device: CPU")
        print("   ⚠️  Running on CPU (will be slow!)")
    
    # Load tokenizer and model
    print(f"\n📦 Loading model: {args.model_name}")
    
    # Try fast tokenizer first, fall back to slow if it fails
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    except Exception as e:
        print(f"⚠️  Fast tokenizer failed: {e}")
        print(f"   Falling back to slow tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set dtype based on device
    # Default to FP32 for stability - FP16 can cause gradient issues
    if args.use_fp16 and device == "cuda":
        dtype = torch.float16
        use_fp16 = True
        print(f"   Using FP16 (half precision)")
    else:
        dtype = torch.float32
        use_fp16 = False
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    
    # Move to device
    model = model.to(device)
    
    print(f"✅ Model loaded!")
    print(f"   Parameters: {model.num_parameters():,}")
    print(f"   Memory: {model.get_memory_footprint() / 1e9:.2f} GB")
    print(f"   Dtype: {dtype}")
    
    # Apply torch.compile for speedup (PyTorch 2.0+)
    # Note: Only enabled for CUDA - MPS support is experimental
    # torch.compile doesn't work well with fp16, so we'll use fp32
    if args.use_compile and device == "cuda":
        try:
            print(f"\n⚡ Compiling model with torch.compile...")
            # Disable fp16 when using compile to avoid gradient scaling issues
            use_fp16 = False
            model = torch.compile(model, mode=args.compile_mode)
            print(f"✅ Model compiled! (mode: {args.compile_mode})")
            print(f"   Expected speedup: 20-30% on T4 GPU")
            print(f"   Note: Using FP32 for compatibility with torch.compile")
        except Exception as e:
            print(f"⚠️  torch.compile failed: {e}")
            print(f"   Continuing without compilation...")
    elif args.use_compile and device == "mps":
        print(f"\n⚠️  torch.compile is experimental on MPS, skipping...")
        print(f"   (Only CUDA is fully supported)")
    
    # Load dataset
    dataset = load_jsonl_dataset(args.dataset_path)
    
    # Tokenize dataset
    print(f"\n🔤 Tokenizing dataset...")
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    
    print("✅ Tokenization complete!")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Training arguments
    print(f"\n⚙️  Training configuration:")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Output dir: {args.output_dir}")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        
        # Optimization
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        
        # Logging
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        
        # Performance - use fp16 only on CUDA
        fp16=use_fp16,
        dataloader_num_workers=0 if device == "cpu" else 2,
        
        # Other
        report_to="none",  # Disable wandb for testing
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,  # Important for causal LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )
    
    # Train!
    print(f"\n🚀 Starting training...")
    print(f"   Estimated time: {estimate_training_time(args, len(dataset['train']), device)}")
    print()
    
    try:
        trainer.train()
        
        print("\n✅ Training complete!")
        
        # Save final model
        final_output_dir = os.path.join(args.output_dir, "final")
        print(f"\n💾 Saving final model to {final_output_dir}...")
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        
        print("✅ Model saved!")
        print(f"\n📊 Training metrics:")
        print(f"   Final train loss: {trainer.state.log_history[-2]['loss']:.4f}")
        print(f"   Final eval loss: {trainer.state.log_history[-1]['eval_loss']:.4f}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("   Saving checkpoint...")
        trainer.save_model(os.path.join(args.output_dir, "interrupted"))
        print("   Checkpoint saved!")
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


def estimate_training_time(args, num_samples, device):
    """Estimate training time"""
    steps_per_epoch = num_samples // args.batch_size
    total_steps = steps_per_epoch * args.num_epochs
    
    # Rough estimates per device
    if device == "cuda":
        seconds_per_step = 0.5  # T4 GPU
    elif device == "mps":
        seconds_per_step = 1.0  # Apple Silicon - faster than CPU, slower than CUDA
    else:
        seconds_per_step = 5.0  # CPU - very slow
    
    total_seconds = total_steps * seconds_per_step
    hours = total_seconds / 3600
    
    if hours < 1:
        return f"{total_seconds / 60:.0f} minutes"
    else:
        return f"{hours:.1f} hours"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen-0.5B for commit messages")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="muellerzr/qwen-0.5-git-commit-message-generation",
        help="Pretrained model name or path"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/formatted",
        help="Path to formatted dataset"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen-0.5b-finetuned",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1 for testing)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        help="Use torch.compile for speedup (PyTorch 2.0+, recommended for T4 GPU)"
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: default, max-autotune for best performance)"
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use FP16 (half precision) training. Faster but can be unstable. Default: FP32"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training
    main(args)
