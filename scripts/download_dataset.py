#!/usr/bin/env python3
"""
Download CommitPackFT dataset from Hugging Face
"""
import os
from datasets import load_dataset
from tqdm import tqdm

def download_commitpackft(output_dir="./data/raw", max_samples=100000, languages=None):
    """
    Download CommitPackFT dataset
    
    Args:
        output_dir: Directory to save the dataset
        max_samples: Maximum number of samples to download
        languages: List of programming languages to download (default: top 5)
    """
    if languages is None:
        # Focus on top programming languages
        languages = ['python', 'javascript', 'java', 'typescript', 'go']
    
    print("📥 Downloading CommitPackFT dataset...")
    print(f"Languages: {', '.join(languages)}")
    print(f"Target: {max_samples:,} samples total")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect samples from each language
    all_samples = []
    samples_per_language = max_samples // len(languages)
    
    for lang in languages:
        print(f"\n📚 Downloading {lang} samples...")
        
        try:
            # Load dataset for this language in streaming mode
            dataset = load_dataset(
                "bigcode/commitpackft",
                lang,
                split="train",
                streaming=True,
            )
            
            # Collect samples for this language
            lang_samples = []
            for i, sample in enumerate(tqdm(dataset, total=samples_per_language, desc=f"{lang}")):
                lang_samples.append(sample)
                
                if len(lang_samples) >= samples_per_language:
                    break
            
            all_samples.extend(lang_samples)
            print(f"  ✓ Downloaded {len(lang_samples):,} {lang} samples")
            
        except Exception as e:
            print(f"  ⚠️  Error downloading {lang}: {e}")
            continue
    
    print(f"\n✅ Total samples collected: {len(all_samples):,}")


    
    print(f"\n✅ Total samples collected: {len(all_samples):,}")
    
    # Convert to Dataset and save
    from datasets import Dataset
    final_dataset = Dataset.from_list(all_samples)
    
    # Save to disk
    output_path = os.path.join(output_dir, "commitpackft")
    final_dataset.save_to_disk(output_path)
    
    print(f"📁 Saved to: {output_path}")
    print(f"💾 Size: {len(all_samples) * 2 / 1000:.1f} MB (estimated)")
    
    return output_path


def main():
    """Main function"""
    print("=" * 50)
    print("CommitPackFT Dataset Downloader")
    print("=" * 50)
    
    # Download dataset from top 5 languages
    dataset_path = download_commitpackft(
        output_dir="./data/raw",
        max_samples=100000,  # 100K samples total (20K per language)
        languages=['python', 'javascript', 'java', 'typescript', 'go']
    )
    
    print("\n✅ Download complete!")
    print(f"\nNext step: Run prepare_dataset.py to clean and filter the data")


if __name__ == "__main__":
    main()
