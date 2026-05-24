#!/usr/bin/env python3
"""
Debug script to test model loading.
Run: python test_model.py
"""

import sys
from pathlib import Path

# Add root to path
root = Path(__file__).parent
sys.path.insert(0, str(root))

from brain.inference import KokomiInference
from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    print("=" * 70)
    print("KOKOMI MODEL LOADING TEST")
    print("=" * 70)
    print()
    
    # Load settings
    settings = Settings.load()
    logger.info("Settings loaded")
    
    # Create inference engine
    inference = KokomiInference(settings=settings)
    logger.info("Inference engine initialized")
    
    # Find latest checkpoint
    latest = inference.find_latest_checkpoint()
    if not latest:
        logger.error("No checkpoint found!")
        print("\nNo checkpoints in:", settings.checkpoint_dir)
        return False
    
    logger.info(f"Latest checkpoint: {latest}")
    
    # Try to load
    print("\nAttempting to load model...")
    success = inference.load(latest)
    
    if success:
        print("\n✓ Model loaded successfully!")
        print(f"  Is loaded: {inference.is_loaded()}")
        print(f"  Model type: {type(inference.model)}")
        print(f"  Tokenizer vocab size: {inference.tokenizer.vocab_size}")
        
        # Try a test generation
        print("\nTesting generation...")
        prompt = "Hello, I am"
        response = inference.generate(prompt, max_tokens=30)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        return True
    else:
        print("\n✗ Failed to load model")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.exception("Error during test")
        sys.exit(1)
