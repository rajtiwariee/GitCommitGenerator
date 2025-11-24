"""
CLI interface for commit-gen
"""
import argparse
import sys
from pathlib import Path

from commit_gen.generator import CommitMessageGenerator
from commit_gen.git_utils import GitRepository
from commit_gen.config import Config


def cmd_generate(args):
    """Generate commit message command"""
    config = Config()
    repo = GitRepository()
    
    # Check if in a git repository
    if not repo.is_git_repo():
        print("Error: Not a Git repository")
        print("Run this command inside a Git repository")
        return 1
    
    # Get diff
    if args.files:
        diff = repo.get_diff_for_files(args.files, staged=args.staged)
    elif args.staged or not args.unstaged:
        diff = repo.get_staged_diff()
    else:
        diff = repo.get_unstaged_diff()
    
    if not diff.strip():
        print("No changes to commit")
        if args.staged:
            print("Stage some changes with: git add <files>")
        return 1
    
    # Format diff for model
    formatted_diff = repo.format_diff_for_model(diff)
    
    # Load model and generate
    model_path = args.model_path or config.get("model_path")
    generator = CommitMessageGenerator(model_path)
    
    print("-" * 60)
    print("Generating commit message...")
    print("-" * 60)
    
    try:
        message = generator.generate(
            formatted_diff,
            temperature=args.temperature or config.get("temperature"),
            max_length=args.max_length or config.get("max_length"),
            top_p=config.get("top_p"),
        )
        
        print(f"\nGenerated Message:")
        print(f"  {message}\n")
        
        # Commit or show interactive prompt
        if args.commit:
            if args.interactive:
                print("Edit message (press Enter to accept, Ctrl+C to cancel):")
                try:
                    edited = input(f"  {message}\n  > ").strip()
                    if edited:
                        message = edited
                except KeyboardInterrupt:
                    print("\nCommit cancelled")
                    return 1
            
            print("Creating commit...")
            repo.create_commit(message)
            print("Commit created successfully!")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_test(args):
    """Test command for custom diffs"""
    config = Config()
    
    # Read diff from file or stdin
    if args.diff_file:
        with open(args.diff_file, 'r') as f:
            diff = f.read()
    else:
        print("Paste your diff (press Ctrl+D when done):")
        diff = sys.stdin.read()
    
    if not diff.strip():
        print("Error: Empty diff provided")
        return 1
    
    # Load model and generate
    model_path = args.model_path or config.get("model_path")
    generator = CommitMessageGenerator(model_path)
    
    print("-" * 60)
    print("Generating commit message...")
    print("-" * 60)
    
    try:
        message = generator.generate(
            diff,
            temperature=args.temperature or config.get("temperature"),
            max_length=args.max_length or config.get("max_length"),
        )
        
        print(f"\nGenerated Message:")
        print(f"  {message}\n")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_config(args):
    """Configuration command"""
    config = Config()
    
    if args.show:
        print("-" * 60)
        print("Current Configuration:")
        print("-" * 60)
        for key, value in config.show().items():
            print(f"  {key}: {value}")
        return 0
    
    # Set configuration values
    if args.model_path:
        config.set("model_path", args.model_path)
        print(f"Set model_path = {args.model_path}")
    
    if args.temperature is not None:
        config.set("temperature", args.temperature)
        print(f"Set temperature = {args.temperature}")
    
    if args.max_length is not None:
        config.set("max_length", args.max_length)
        print(f"Set max_length = {args.max_length}")
    
    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="🤖 AI-powered Git commit message generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate commit message for changes",
        aliases=['gen', 'g'],
    )
    gen_parser.add_argument(
        "--files",
        nargs="+",
        help="Specific files to generate message for"
    )
    gen_parser.add_argument(
        "--staged",
        action="store_true",
        default=True,
        help="Use staged changes (default)"
    )
    gen_parser.add_argument(
        "--unstaged",
        action="store_true",
        help="Use unstaged changes"
    )
    gen_parser.add_argument(
        "--commit",
        "-c",
        action="store_true",
        help="Automatically create commit with generated message"
    )
    gen_parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Edit message before committing (requires --commit)"
    )
    gen_parser.add_argument(
        "--model-path",
        help="Path to fine-tuned model (overrides config)"
    )
    gen_parser.add_argument(
        "--temperature",
        type=float,
        help="Generation temperature"
    )
    gen_parser.add_argument(
        "--max-length",
        type=int,
        help="Maximum message length"
    )
    gen_parser.set_defaults(func=cmd_generate)
    
    # Test command
    test_parser = subparsers.add_parser(
        "test",
        help="Test on custom diff",
    )
    test_parser.add_argument(
        "--diff-file",
        help="File containing diff (default: stdin)"
    )
    test_parser.add_argument(
        "--model-path",
        help="Path to fine-tuned model"
    )
    test_parser.add_argument(
        "--temperature",
        type=float,
        help="Generation temperature"
    )
    test_parser.add_argument(
        "--max-length",
        type=int,
        help="Maximum message length"
    )
    test_parser.set_defaults(func=cmd_test)
    
    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Manage configuration",
    )
    config_parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration"
    )
    config_parser.add_argument(
        "--model-path",
        help="Set model path"
    )
    config_parser.add_argument(
        "--temperature",
        type=float,
        help="Set generation temperature"
    )
    config_parser.add_argument(
        "--max-length",
        type=int,
        help="Set maximum message length"
    )
    config_parser.set_defaults(func=cmd_config)
    
    # Parse and run
    args = parser.parse_args()
    
    if not args.command:
        # Default to generate command
        args = parser.parse_args(['generate'])
    
    if hasattr(args, 'func'):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
