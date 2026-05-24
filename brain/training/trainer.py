"""
Training pipeline — upgraded for BPE tokenizer and multi-file datasets.
"""

import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from brain.model.transformer import KokomiTransformer
from brain.tokenizer.tokenizer import BPETokenizer
from brain.training.dataset import TextDataset
from brain.training.loss import cross_entropy_loss
from config.settings import Settings
from utils.logger import get_logger
from utils.helpers import get_device, count_parameters, format_number, ensure_dir, timestamp

logger = get_logger(__name__)


class Trainer:
    def __init__(self, settings: Settings = None):
        self.settings  = settings or Settings.load()
        self.device    = get_device()
        self.model     = None
        self.tokenizer = None

    def _load_all_text(self) -> str:
        """Load and merge ALL .txt files from data/raw/"""
        raw_dir = Path(self.settings.data_dir) / "raw"
        files   = sorted(raw_dir.glob("*.txt"))

        if not files:
            logger.warning("No .txt files found in data/raw/. Creating sample.")
            sample = raw_dir / "train.txt"
            ensure_dir(raw_dir)
            sample.write_text(
                "Hello, I am Kokomi. I help you build, think, and create.\n" * 200,
                encoding="utf-8"
            )
            files = [sample]

        parts = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                parts.append(text)
                logger.info(f"Loaded {f.name}: {len(text):,} chars")
            except Exception as e:
                logger.warning(f"Could not read {f.name}: {e}")

        combined = "\n\n".join(parts)
        logger.info(f"Total training text: {len(combined):,} chars from {len(files)} file(s)")
        return combined

    def setup(self):
        cfg  = self.settings
        text = self._load_all_text()

        # BPE tokenizer — smarter than char-level
        self.tokenizer = BPETokenizer(vocab_size=cfg.model.vocab_size)
        self.tokenizer.build(text)

        # Save tokenizer alongside checkpoints for inference
        tok_path = Path(cfg.checkpoint_dir) / "tokenizer.json"
        ensure_dir(cfg.checkpoint_dir)
        self.tokenizer.save(str(tok_path))

        # Update vocab size to match actual built vocab
        cfg.model.vocab_size = self.tokenizer.vocab_size

        # Build model
        self.model = KokomiTransformer.from_config(cfg.model).to(self.device)
        logger.info(f"Model: {format_number(count_parameters(self.model))} parameters on {self.device}")

        # Dataset
        dataset    = TextDataset(text, self.tokenizer, cfg.model.max_seq_len)
        val_size   = max(1, int(0.1 * len(dataset)))
        train_size = len(dataset) - val_size
        self.train_set, self.val_set = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            self.train_set,
            batch_size  = cfg.training.batch_size,
            shuffle     = True,
            pin_memory  = (self.device.type == "cuda"),
            num_workers = 0,
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size  = cfg.training.batch_size,
            shuffle     = False,
            num_workers = 0,
        )
        logger.info(f"Train: {len(self.train_set):,} | Val: {len(self.val_set):,}")

    def train_epoch(self, optimizer, epoch: int) -> float:
        self.model.train()
        cfg   = self.settings.training
        total = 0.0
        steps = 0

        for step, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            logits = self.model(x)
            loss   = cross_entropy_loss(logits, y, self.tokenizer.pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            optimizer.step()
            total += loss.item()
            steps += 1

            if step % cfg.checkpoint_every == 0 and step > 0:
                self._save_checkpoint(epoch, step)

        return total / max(steps, 1)

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total, steps = 0.0, 0
        for x, y in self.val_loader:
            x, y   = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss   = cross_entropy_loss(logits, y, self.tokenizer.pad_id)
            total += loss.item()
            steps += 1
        return total / max(steps, 1)

    def run(self):
        self.setup()
        cfg       = self.settings.training
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = cfg.learning_rate,
            weight_decay = 0.01,
        )
        # Cosine LR scheduler — decays learning rate smoothly
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=1e-5
        )

        logger.info(f"Training on {self.device} for {cfg.epochs} epochs")
        for epoch in range(1, cfg.epochs + 1):
            train_loss = self.train_epoch(optimizer, epoch)
            val_loss   = self.evaluate()
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch}/{cfg.epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | lr={lr:.2e}"
            )

        self._save_checkpoint("final")
        logger.info("Training complete.")

    def _save_checkpoint(self, epoch, step: int = 0):
        ensure_dir(self.settings.checkpoint_dir)
        path = Path(self.settings.checkpoint_dir) / f"kokomi_e{epoch}_s{step}_{timestamp()}.pt"
        torch.save({
            "epoch":      epoch,
            "step":       step,
            "model":      self.model.state_dict(),
            "tokenizer":  "bpe",   # signal to inference: load tokenizer.json
            "config":     self.settings.model.__dict__,
        }, path)
        logger.info(f"Checkpoint saved: {path.name}")
