"""
Configuration management for commit-gen CLI
"""
import os
import json
from pathlib import Path


class Config:
    """Configuration manager"""
    
    DEFAULT_CONFIG = {
        "model_path": "./qwen-0.5b-finetuned/final",
        "temperature": 0.7,
        "max_length": 100,
        "top_p": 0.9,
    }
    
    def __init__(self):
        self.config_dir = Path.home() / ".config" / "commit-gen"
        self.config_file = self.config_dir / "config.json"
        self.config = self.load()
    
    def load(self):
        """Load configuration from file"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return {**self.DEFAULT_CONFIG, **json.load(f)}
        return self.DEFAULT_CONFIG.copy()
    
    def save(self):
        """Save configuration to file"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set configuration value"""
        self.config[key] = value
        self.save()
    
    def show(self):
        """Show all configuration"""
        return self.config
