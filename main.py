"""
KOKOMI - Local AI Assistant
Entry point. Run this to start.

Usage:
    python main.py            # CLI mode
    python main.py --web      # Web dashboard mode
    python main.py --train    # Train the model
"""

import argparse
from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="KOKOMI — Local AI Assistant")
    parser.add_argument("--web",   action="store_true", help="Start web dashboard")
    parser.add_argument("--train", action="store_true", help="Run training pipeline")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def run_cli():
    from ui.cli.terminal import Terminal
    terminal = Terminal()
    terminal.run()


def run_web():
    from ui.web.server import WebServer
    server = WebServer()
    server.run()


def run_training():
    from brain.training.trainer import Trainer
    trainer = Trainer()
    trainer.run()


def main():
    args = parse_args()
    settings = Settings.load()

    if args.debug:
        settings.debug = True

    logger.info("KOKOMI starting up...")

    if args.train:
        run_training()
    elif args.web:
        run_web()
    else:
        run_cli()


if __name__ == "__main__":
    main()
