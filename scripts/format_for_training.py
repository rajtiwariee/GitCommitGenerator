#!/usr/bin/env python3
"""
Format dataset for model training (instruction format)
"""
import os
import json
from tqdm import tqdm


def format_sample(message, diff):
    """
    Format a single sample in instruction format
    
    Returns formatted text ready for training
    """
    # Simple instruction format (adjust based on what works best)
    formatted_text = f"""Write a git commit message:

Diff:
{diff}

Commit message:
{message}"""
    
    return formatted_text


def format_sample_chat_template(message, diff):
    """
    Alternative: Format with chat template (Qwen2 style)
    """
    formatted_text = f"""<|im_start|>system
You are an expert at writing clear, concise git commit messages.<|im_end|>
<|im_start|>user
Generate a commit message for these code changes:

{diff}<|im_end|>
<|im_start|>assistant
{message}<|im_end|>"""
    
    return formatted_text


def format_dataset(input_dir, output_dir, use_chat_template=False):
    """
    Format dataset for training
    
    Args:
        input_dir: Directory with processed data (*.jsonl files)
        output_dir: Directory to save formatted data  
        use_chat_template: Whether to use chat template format
    """
    print("📝 Formatting dataset for training...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        input_file = os.path.join(input_dir, f"{split}.jsonl")
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        
        if not os.path.exists(input_file):
            print(f"⚠️  Skipping {split} (file not found)")
            continue
        
        print(f"\n📄 Processing {split}...")
        
        # Read input
        samples = []
        with open(input_file, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
        
        # Format samples
        formatted_samples = []
        for sample in tqdm(samples):
            message = sample['message']
            diff = sample['diff']
            
            # Choose formatting function
            if use_chat_template:
                text = format_sample_chat_template(message, diff)
            else:
                text = format_sample(message, diff)
            
            formatted_samples.append({'text': text})
        
        # Save formatted data
        with open(output_file, 'w') as f:
            for item in formatted_samples:
                f.write(json.dumps(item) + '\n')
        
        print(f"  ✓ {split}: {len(formatted_samples):,} samples → {output_file}")
    
    print(f"\n✅ Dataset formatted!")
    print(f"📁 Saved to: {output_dir}")


def main():
    """Main function"""
    print("=" * 50)
    print("Dataset Formatting for Training")
    print("=" * 50)
    
    # Paths
    input_dir = "./data/processed"
    output_dir = "./data/formatted"
    
    # Format dataset
    # Try simple format first, can switch to chat template later
    format_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        use_chat_template=False  # Set to True if simple format doesn't work
    )
    
    print(f"\n✅ Ready for training!")
    print(f"\nNext step: Run training script")
    print(f"  python train.py --dataset_path ./data/formatted")


if __name__ == "__main__":
    main()
