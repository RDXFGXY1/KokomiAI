#!/usr/bin/env python3
"""
Quick demo script to test the chat interface.
Run: python demo_chat.py
"""

import sys
from pathlib import Path

# Add root to path so imports work
root = Path(__file__).parent
sys.path.insert(0, str(root))

from ui.cli.terminal import Terminal
from config.settings import Settings

def main():
    print("=" * 60)
    print("KOKOMI AI ASSISTANT - CLI CHAT DEMO")
    print("=" * 60)
    print()
    
    # Initialize settings
    settings = Settings.load()
    
    # Create and run terminal
    terminal = Terminal(settings=settings)
    
    print("\nStarting chat interface...")
    print("Tip: Type /help to see all commands\n")
    
    terminal.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nChat interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
