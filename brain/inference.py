"""
Model inference engine — upgraded for BPE tokenizer.
"""

import torch
from pathlib import Path
from typing import Optional
from brain.model.transformer import KokomiTransformer
from brain.tokenizer.tokenizer import BPETokenizer, CharTokenizer
from config.settings import Settings
from utils.logger import get_logger
from utils.helpers import get_device

logger = get_logger(__name__)

PROMPT_TEMPLATE = "RDXFGXY1: {input}\nKokomi:"


class KokomiInference:
    def __init__(self, checkpoint_path: Optional[str] = None, settings: Settings = None):
        self.settings     = settings or Settings.load()
        self.device       = get_device()
        self.model        = None
        self.tokenizer    = None
        self._max_seq_len = 256

        if checkpoint_path:
            self.load(checkpoint_path)

    def load(self, checkpoint_path: str) -> bool:
        path = Path(checkpoint_path)
        if not path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            cfg = checkpoint.get("config", {})
            self._max_seq_len = cfg.get("max_seq_len", 256)

            # Load tokenizer — BPE from file or fallback to char from checkpoint
            tok_type = checkpoint.get("tokenizer", "char")
            if tok_type == "bpe":
                tok_path = path.parent / "tokenizer.json"
                if tok_path.exists():
                    self.tokenizer = BPETokenizer.load(str(tok_path))
                else:
                    logger.error("BPE checkpoint but tokenizer.json not found.")
                    return False
            else:
                # Legacy char tokenizer
                self.tokenizer = CharTokenizer()
                self.tokenizer.char_to_id = tok_type if isinstance(tok_type, dict) else checkpoint["tokenizer"]
                self.tokenizer.id_to_char = {v: k for k, v in self.tokenizer.char_to_id.items()}
                self.tokenizer._built = True

            # Build model from saved config
            self.model = KokomiTransformer(
                vocab_size  = cfg.get("vocab_size",  self.tokenizer.vocab_size),
                embed_dim   = cfg.get("embed_dim",   256),
                num_heads   = cfg.get("num_heads",   8),
                num_layers  = cfg.get("num_layers",  6),
                ff_dim      = cfg.get("ff_dim",      1024),
                max_seq_len = self._max_seq_len,
                dropout     = 0.0,
            )
            self.model.load_state_dict(checkpoint["model"])
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded — vocab: {self.tokenizer.vocab_size} | seq_len: {self._max_seq_len}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None
            return False

    def find_latest_checkpoint(self) -> Optional[str]:
        ckpt_dir = Path(self.settings.checkpoint_dir)
        if not ckpt_dir.exists():
            return None
        checkpoints = list(ckpt_dir.glob("*.pt"))
        if not checkpoints:
            return None
        return str(sorted(checkpoints)[-1])

    @torch.no_grad()
    def generate(
        self,
        user_input:  str,
        max_tokens:  int   = 150,
        temperature: float = 0.7,
        top_k:       int   = 40,
    ) -> str:
        if not self.model or not self.tokenizer:
            return "[Model not loaded]"

        prompt     = PROMPT_TEMPLATE.format(input=user_input.strip())
        prompt_ids = self.tokenizer.encode(prompt)
        generated  = list(prompt_ids)
        new_tokens = []

        for _ in range(max_tokens):
            window = generated[-self._max_seq_len:]
            x      = torch.tensor([window], dtype=torch.long, device=self.device)

            logits     = self.model(x)
            next_logit = logits[0, -1, :] / temperature

            top_vals, top_idx = torch.topk(next_logit, min(top_k, next_logit.size(-1)))
            probs      = torch.softmax(top_vals, dim=-1)
            chosen     = torch.multinomial(probs, 1).item()
            next_id    = top_idx[chosen].item()

            generated.append(next_id)
            new_tokens.append(next_id)

            so_far = self.tokenizer.decode(new_tokens)

            if "\nRDXFGXY1:" in so_far or "\nKokomi:" in so_far:
                so_far = so_far.split("\nRDXFGXY1:")[0].split("\nKokomi:")[0]
                return self._clean(so_far)

            if so_far.endswith("\n\n"):
                return self._clean(so_far)

        return self._clean(self.tokenizer.decode(new_tokens))

    def _clean(self, text: str) -> str:
        text = text.strip()
        for artifact in ["RDXFGXY1:", "Kokomi:", "---"]:
            if text.startswith(artifact):
                text = text[len(artifact):].strip()
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        return text.strip() or "..."

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None
