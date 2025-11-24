#!/usr/bin/env python3
"""
Test the fine-tuned model on the test set and calculate metrics.
Usage: python scripts/test_model.py --model_path ./qwen-0.5b-finetuned/final
"""
import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Import BLEU and ROUGE metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    METRICS_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Install nltk and rouge-score for metrics:")
    print("   pip install nltk rouge-score")
    METRICS_AVAILABLE = False


def load_model_and_tokenizer(model_path, device):
    """Load fine-tuned model and tokenizer"""
    print(f"📦 Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Set dtype based on device
    if device == "cuda":
        dtype = torch.float16
    elif device == "mps":
        dtype = torch.float32  # MPS works better with FP32
    else:
        dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    return model, tokenizer


def extract_commit_message(generated_text, input_text):
    """Extract the commit message from generated text"""
    # Remove the input prompt
    if input_text in generated_text:
        generated_text = generated_text[len(input_text):].strip()
    
    # Take only the first line
    lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
    if not lines:
        return generated_text.strip()
    
    commit_msg = lines[0]
    
    # Remove common artifacts
    # Remove dates/timestamps (e.g., "2017-03-13", "(2)")
    import re
    commit_msg = re.sub(r'\s*\(\d+\)\s*-\s*\d{4}-\d{2}-\d{2}.*$', '', commit_msg)
    commit_msg = re.sub(r'\s*-\s*\d{4}-\d{2}-\d{2}.*$', '', commit_msg)
    commit_msg = re.sub(r'\s*\(\d+\).*$', '', commit_msg)
    
    # Remove trailing punctuation artifacts
    commit_msg = re.sub(r'\.\s*\(.*$', '.', commit_msg)
    
    # Limit length (commit messages should be ~50-72 chars)
    if len(commit_msg) > 100:
        # Try to cut at a natural boundary
        commit_msg = commit_msg[:100].rsplit(' ', 1)[0]
    
    return commit_msg.strip()


def generate_commit_message(model, tokenizer, diff_text, device, max_new_tokens=30):
    """Generate a commit message for a given diff"""
    # Create prompt (matching training format)
    prompt = f"Write a git commit message:\n\n{diff_text}\n\nCommit message:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,  # Very low temperature for focused output
            do_sample=True,
            top_p=0.85,
            repetition_penalty=1.2,  # Penalize repetition
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    commit_msg = extract_commit_message(generated_text, prompt)
    
    return commit_msg


def calculate_bleu(reference, hypothesis):
    """Calculate BLEU score"""
    if not METRICS_AVAILABLE:
        return 0.0
    
    # Tokenize
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    # Use smoothing for short sentences
    smoothing = SmoothingFunction().method1
    score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
    
    return score


def calculate_rouge(reference, hypothesis):
    """Calculate ROUGE scores"""
    if not METRICS_AVAILABLE:
        return {}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }


def evaluate_model(model, tokenizer, test_dataset, device, num_samples=None):
    """Evaluate model on test set"""
    print(f"\n🔍 Evaluating model on test set...")
    
    if num_samples:
        test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))
    
    results = {
        'bleu_scores': [],
        'rouge_scores': defaultdict(list),
        'exact_matches': 0,
        'examples': [],
    }
    
    for i, example in enumerate(tqdm(test_dataset, desc="Evaluating")):
        # Extract diff from the formatted text
        text = example['text']
        
        # Parse the formatted text to get diff and reference message
        if "Diff:" not in text or "Commit message:" not in text:
            continue
            
        diff_start = text.index("Diff:")
        msg_start = text.index("Commit message:")
        
        diff_text = text[diff_start:msg_start].strip()
        reference_msg = text[msg_start + len("Commit message:"):].strip()
        
        # Generate prediction
        predicted_msg = generate_commit_message(model, tokenizer, diff_text, device)
        
        # Calculate metrics
        bleu = calculate_bleu(reference_msg, predicted_msg)
        rouge = calculate_rouge(reference_msg, predicted_msg)
        exact_match = (predicted_msg.lower() == reference_msg.lower())
        
        results['bleu_scores'].append(bleu)
        for key, value in rouge.items():
            results['rouge_scores'][key].append(value)
        if exact_match:
            results['exact_matches'] += 1
        
        # Save first 10 examples
        if i < 10:
            results['examples'].append({
                'diff': diff_text[:200] + '...',  # Truncate for readability
                'reference': reference_msg,
                'predicted': predicted_msg,
                'bleu': bleu,
                'rouge1': rouge.get('rouge1', 0),
            })
    
    # Calculate averages
    results['avg_bleu'] = np.mean(results['bleu_scores'])
    results['avg_rouge'] = {
        key: np.mean(values) for key, values in results['rouge_scores'].items()
    }
    results['exact_match_rate'] = results['exact_matches'] / len(test_dataset)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned model")
    parser.add_argument("--model_path", type=str, default="./qwen-0.5b-finetuned/final", help="Path to fine-tuned model")
    parser.add_argument("--dataset_path", type=str, default="./data/formatted", help="Path to formatted dataset")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output JSON file")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of test samples (default: all)")
    args = parser.parse_args()
    
    # Check device - prioritize MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"🖥️  Device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"🖥️  Device: CUDA GPU")
    else:
        device = "cpu"
        print(f"🖥️  Device: CPU (⚠️  This will be slow!)")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    
    # Load test dataset
    print(f"\n📂 Loading test dataset from {args.dataset_path}...")
    dataset = load_dataset('json', data_files={
        'test': os.path.join(args.dataset_path, 'test.jsonl')
    })
    test_dataset = dataset['test']
    print(f"✅ Loaded {len(test_dataset)} test samples")
    
    # Evaluate
    results = evaluate_model(model, tokenizer, test_dataset, device, args.num_samples)
    
    # Print results
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"Average BLEU Score: {results['avg_bleu']:.4f}")
    if results['avg_rouge']:
        print(f"Average ROUGE-1: {results['avg_rouge']['rouge1']:.4f}")
        print(f"Average ROUGE-2: {results['avg_rouge']['rouge2']:.4f}")
        print(f"Average ROUGE-L: {results['avg_rouge']['rougeL']:.4f}")
    print(f"Exact Match Rate: {results['exact_match_rate']:.2%}")
    
    print("\n📝 Example Predictions:")
    for i, ex in enumerate(results['examples'][:5], 1):
        print(f"\nExample {i}:")
        print(f"  Reference: {ex['reference']}")
        print(f"  Predicted: {ex['predicted']}")
        print(f"  BLEU: {ex['bleu']:.4f}")
    
    # Save results
    # Convert numpy types to Python types for JSON
    results_serializable = {
        'avg_bleu': float(results['avg_bleu']),
        'avg_rouge': {k: float(v) for k, v in results['avg_rouge'].items()},
        'exact_match_rate': float(results['exact_match_rate']),
        'examples': results['examples'],
    }
    
    with open(args.output, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
