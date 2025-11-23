#!/usr/bin/env python3
"""
Script to plot training and validation loss from Hugging Face Trainer checkpoints.
Usage: python scripts/plot_loss.py --model_dir ./qwen-0.5b-finetuned
"""
import os
import json
import argparse
import matplotlib.pyplot as plt
import glob
import pandas as pd

def parse_trainer_state(checkpoint_dir):
    """Parse trainer_state.json from a checkpoint directory"""
    state_file = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.exists(state_file):
        return []
    
    with open(state_file, 'r') as f:
        data = json.load(f)
    
    return data.get("log_history", [])

def main():
    parser = argparse.ArgumentParser(description="Plot training loss")
    parser.add_argument("--model_dir", type=str, default="./qwen-0.5b-finetuned", help="Directory containing checkpoints")
    parser.add_argument("--output", type=str, default="loss_plot.png", help="Output image file")
    args = parser.parse_args()
    
    # Find all checkpoints
    checkpoints = glob.glob(os.path.join(args.model_dir, "checkpoint-*"))
    if not checkpoints:
        print(f"No checkpoints found in {args.model_dir}")
        return

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest_checkpoint = checkpoints[-1]
    print(f"Reading logs from latest checkpoint: {latest_checkpoint}")
    
    history = parse_trainer_state(latest_checkpoint)
    if not history:
        print("No log history found.")
        return
    
    # Extract data
    train_loss = []
    eval_loss = []
    
    for entry in history:
        if "loss" in entry:
            train_loss.append({"step": entry["step"], "loss": entry["loss"]})
        if "eval_loss" in entry:
            eval_loss.append({"step": entry["step"], "loss": entry["eval_loss"]})
            
    train_df = pd.DataFrame(train_loss)
    eval_df = pd.DataFrame(eval_loss)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    if not train_df.empty:
        plt.plot(train_df["step"], train_df["loss"], label="Training Loss", alpha=0.6)
        # Add moving average for smoother training curve
        if len(train_df) > 10:
            plt.plot(train_df["step"], train_df["loss"].rolling(10).mean(), label="Train Loss (Smoothed)", color='blue', linewidth=2)
            
    if not eval_df.empty:
        plt.plot(eval_df["step"], eval_df["loss"], label="Validation Loss", marker='o', color='red', linewidth=2)
    
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(args.output)
    print(f"✅ Loss plot saved to {args.output}")

if __name__ == "__main__":
    main()
