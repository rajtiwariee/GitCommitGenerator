"""
Git repository utilities for commit-gen CLI
"""
import subprocess
import os
from typing import List, Optional
from pathlib import Path


class GitRepository:
    """Interface for Git operations"""
    
    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize Git repository interface
        
        Args:
            repo_path: Path to Git repository (default: current directory)
        """
        self.repo_path = repo_path or os.getcwd()
    
    def is_git_repo(self) -> bool:
        """Check if current directory is a Git repository"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def get_staged_diff(self) -> str:
        """
        Get diff of staged changes
        
        Returns:
            Diff text of staged changes
        """
        result = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    
    def get_unstaged_diff(self) -> str:
        """
        Get diff of unstaged changes
        
        Returns:
            Diff text of unstaged changes
        """
        result = subprocess.run(
            ["git", "diff"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    
    def get_diff_for_files(self, files: List[str], staged: bool = True) -> str:
        """
        Get diff for specific files
        
        Args:
            files: List of file paths
            staged: If True, get staged diff; otherwise unstaged
        
        Returns:
            Diff text for specified files
        """
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--cached")
        cmd.extend(["--"] + files)
        
        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    
    def has_staged_changes(self) -> bool:
        """Check if there are any staged changes"""
        diff = self.get_staged_diff()
        return bool(diff.strip())
    
    def create_commit(self, message: str, allow_empty: bool = False):
        """
        Create a Git commit with the given message
        
        Args:
            message: Commit message
            allow_empty: Allow empty commits
        """
        cmd = ["git", "commit", "-m", message]
        if allow_empty:
            cmd.append("--allow-empty")
        
        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Git commit failed: {result.stderr}")
        
        return result.stdout
    
    def get_status(self) -> str:
        """Get Git status"""
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    
    def format_diff_for_model(self, diff_text: str) -> str:
        """
        Format raw git diff into the model's expected format
        
        Args:
            diff_text: Raw git diff output
        
        Returns:
            Formatted diff text
        """
        # For now, we'll use a simplified format
        # In the future, we can parse the diff more intelligently
        
        if not diff_text.strip():
            return "No changes"
        
        # Extract file information and changes
        lines = diff_text.split('\n')
        current_file = None
        old_content = []
        new_content = []
        language = "Unknown"
        
        formatted_parts = []
        
        for line in lines:
            if line.startswith('diff --git'):
                # New file, save previous if exists
                if current_file:
                    formatted_parts.append(self._format_file_diff(
                        current_file, language, old_content, new_content
                    ))
                    old_content = []
                    new_content = []
                
                # Extract filename
                parts = line.split()
                if len(parts) >= 4:
                    current_file = parts[3].lstrip('b/')
                    language = self._detect_language(current_file)
            
            elif line.startswith('-') and not line.startswith('---'):
                old_content.append(line[1:])
            elif line.startswith('+') and not line.startswith('+++'):
                new_content.append(line[1:])
        
        # Add last file
        if current_file:
            formatted_parts.append(self._format_file_diff(
                current_file, language, old_content, new_content
            ))
        
        return '\n\n'.join(formatted_parts) if formatted_parts else diff_text
    
    def _format_file_diff(self, file_path, language, old_content, new_content):
        """Format a single file diff"""
        return f"""Diff:
File: {file_path}
Language: {language}

Old content:
{chr(10).join(old_content[:20])}  # Limit to 20 lines

New content:
{chr(10).join(new_content[:20])}  # Limit to 20 lines"""
    
    def _detect_language(self, file_path):
        """Detect programming language from file extension"""
        ext_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.sh': 'Shell',
            '.md': 'Markdown',
            '.html': 'HTML',
            '.css': 'CSS',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
        }
        
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, 'Unknown')
