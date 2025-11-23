#!/usr/bin/env python3
"""
Prepare and clean the CommitPackFT dataset for training
"""
import os
import re
import json
from datasets import load_from_disk
from tqdm import tqdm


def is_valid_commit_message(message):
    """Check if commit message is valid"""
    # Convert to string and strip
    msg = str(message).strip()
    
    # Length checks
    if len(msg) < 10 or len(msg) > 200:
        return False
    
    # Filter lazy messages
    lazy_patterns = [
        r'^(fix|update|wip|stuff|changes|tmp|test|debug)$',
        r'^(fixed|updated|refactor|refactored)$',
        r'^\.',  # Starts with period
        r'^-',   # Starts with dash
    ]
    
    msg_lower = msg.lower()
    for pattern in lazy_patterns:
        if re.match(pattern, msg_lower):
            return False
    
    # Filter system messages
    system_keywords = ['merge', 'revert', 'automated', 'auto-generated', 'ci:', 'cd:']
    for keyword in system_keywords:
        if keyword in msg_lower:
            return False
    
    return True


def is_valid_diff(old_contents, new_contents):
    """Check if diff is valid"""
    # Check if we have actual content
    if not old_contents and not new_contents:
        return False
    
    # Convert to strings
    old_str = str(old_contents) if old_contents else ""
    new_str = str(new_contents) if new_contents else ""
    
    # Check if there's actually a change
    if old_str == new_str:
        return False
    
    # Check reasonable size (not too large)
    total_lines = len(old_str.split('\n')) + len(new_str.split('\n'))
    if total_lines > 1000:  # Too large
        return False
    
    return True


def filter_by_language(lang, target_languages=None):
    """
    Filter by programming language using the 'lang' field
    
    Args:
        lang: Language string from dataset
        target_languages: List of target languages
    """
    if target_languages is None:
        target_languages = ['Python', 'JavaScript', 'Java', 'TypeScript', 'Go']
    
    if lang is None:
        return False
    
    # Direct match (case-sensitive since dataset uses specific capitalization)
    return str(lang) in target_languages


def prepare_dataset(input_path, output_dir, target_languages=None):
    """
    Clean and filter the dataset
    
    Args:
        input_path: Path to raw dataset
        output_dir: Directory to save cleaned data
        target_languages: List of programming language extensions
    """
    print("🧹 Cleaning and filtering dataset...")
    
    # Load raw dataset
    dataset = load_from_disk(input_path)
    print(f"📊 Loaded {len(dataset):,} samples")
    
    # Filter samples
    filtered_samples = []
    stats = {
        'total': len(dataset),
        'valid_message': 0,
        'valid_diff': 0,
        'valid_language': 0,
        'final': 0,
    }
    
    print("\n⏳ Filtering samples...")
    for sample in tqdm(dataset):
        # Extract fields from CommitPackFT structure
        message = sample.get('subject', '') or sample.get('message', '')
        old_contents = sample.get('old_contents', '')
        new_contents = sample.get('new_contents', '')
        lang = sample.get('lang', '')
        old_file = sample.get('old_file', '')
        new_file = sample.get('new_file', '')
        
        # Apply filters
        if not is_valid_commit_message(message):
            continue
        stats['valid_message'] += 1
        
        if not is_valid_diff(old_contents, new_contents):
            continue
        stats['valid_diff'] += 1
        
        if not filter_by_language(lang, target_languages):
            continue
        stats['valid_language'] += 1
        
        # Create a simple diff representation
        # For training, we'll use a simplified format showing the change
        diff_repr = f"""File: {old_file} -> {new_file}
Language: {lang}

Old content:
{old_contents[:500] if old_contents else '(empty)'}...

New content:
{new_contents[:500] if new_contents else '(empty)'}...
"""
        
        # Keep this sample
        filtered_samples.append({
            'message': message.strip(),
            'diff': diff_repr.strip(),
        })
        stats['final'] += 1
    
    print(f"\n📈 Filtering Statistics:")
    print(f"  Total samples: {stats['total']:,}")
    print(f"  Valid messages: {stats['valid_message']:,} ({stats['valid_message']/stats['total']*100:.1f}%)")
    print(f"  Valid diffs: {stats['valid_diff']:,} ({stats['valid_diff']/stats['total']*100:.1f}%)")
    print(f"  Valid language: {stats['valid_language']:,} ({stats['valid_language']/stats['total']*100:.1f}%)")
    print(f"  Final samples: {stats['final']:,} ({stats['final']/stats['total']*100:.1f}%)")
    
    # Create train/val/test splits
    from datasets import Dataset
    full_dataset = Dataset.from_list(filtered_samples)
    
    # 80% train, 10% val, 10% test
    train_test = full_dataset.train_test_split(test_size=0.2, seed=42)
    val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)
    
    splits = {
        'train': train_test['train'],
        'val': val_test['train'],
        'test': val_test['test'],
    }
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")
        
        with open(output_path, 'w') as f:
            for item in split_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"  ✓ {split_name}: {len(split_data):,} samples → {output_path}")
    
    print(f"\n✅ Dataset prepared!")
    print(f"📁 Saved to: {output_dir}")


def main():
    """Main function"""
    print("=" * 50)
    print("Dataset Preparation")
    print("=" * 50)
    
    # Paths
    input_path = "./data/raw/commitpackft"
    output_dir = "./data/processed"
    
    # Target programming languages (match dataset capitalization!)
    target_languages = ['Python', 'JavaScript', 'Java', 'TypeScript', 'Go']
    
    # Prepare dataset
    prepare_dataset(
        input_path=input_path,
        output_dir=output_dir,
        target_languages=target_languages
    )
    
    print(f"\nNext step: Run format_for_training.py to format data for fine-tuning")


if __name__ == "__main__":
    main()
