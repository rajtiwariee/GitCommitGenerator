"""
Model loading and commit message generation
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class CommitMessageGenerator:
    """Generate commit messages using the fine-tuned model"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the generator
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to use ("cuda", "mps", or "cpu"). Auto-detected if None.
        """
        self.model_path = model_path
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """Load model and tokenizer"""
        if self.model is not None:
            return  # Already loaded
        
        print(f"🔧 Loading model from {self.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded on {self.device}")
    
    def generate(
        self,
        diff: str,
        temperature: float = 0.0,  # Greedy by default for cleaner output
        max_length: int = 20,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a commit message for the given diff
        
        Args:
            diff: Git diff text
            temperature: Sampling temperature (0.0 = deterministic, recommended)
            max_length: Maximum length of generated message
            top_p: Nucleus sampling threshold
        
        Returns:
            Generated commit message
        """
        if self.model is None:
            self.load()
        
        # Create prompt (match training format)
        prompt = f"Write a git commit message:\n\n{diff}\n\nCommit message:\n"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=384
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract commit message
        if prompt in generated_text:
            commit_msg = generated_text[len(prompt):].strip()
        else:
            commit_msg = generated_text.strip()
        
        # Clean up - take first line only
        lines = [line.strip() for line in commit_msg.split('\n') if line.strip()]
        if lines:
            commit_msg = lines[0]
        
        # Remove artifacts (timestamps, issue numbers at the end, etc.)
        import re
        commit_msg = re.sub(r'\s*\(\d+\)\s*-\s*\d{4}-\d{2}-\d{2}.*$', '', commit_msg)
        commit_msg = re.sub(r'\s*-\s*\d{4}-\d{2}-\d{2}.*$', '', commit_msg)
        commit_msg = re.sub(r'\s*Fixes\s+#\d+.*$', '', commit_msg)
        commit_msg = re.sub(r'\s*Closes\s+#\d+.*$', '', commit_msg)
        
        # Cut at first period if it's getting long
        if len(commit_msg) > 80 and '.' in commit_msg:
            commit_msg = commit_msg.split('.')[0] + '.'
        
        # Absolute length limit
        if len(commit_msg) > 100:
            commit_msg = commit_msg[:97] + '...'
        
        return commit_msg.strip()
    
    def unload(self):
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
