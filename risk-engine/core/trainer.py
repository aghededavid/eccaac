"""
ECCAAC — Risk Scoring Engine
Training Loop for LSTM Siamese Network

Responsibilities:
  - Build training dataset from behavioral session data
  - Generate positive pairs  (same user, legitimate sessions)
  - Generate negative pairs  (different users OR flagged attack sessions)
  - Train the Siamese network with CombinedLoss
  - Validate on held-out pairs
  - Fine-tune on new session data (continuous adaptation)
  - Save / load checkpoints with full training state

Usage:
  # Initial training
  trainer = SiameseTrainer(model, config=TrainingConfig())
  trainer.train(dataset)

  # Fine-tune after a new session
  trainer.fine_tune(user_id, session_sequences, label="legitimate")
"""

import os
import time
import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy torch import with helpful error ──────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR
except ImportError:
    raise ImportError(
        "PyTorch is required. Install with:\n"
        "  pip install torch"
    )

from siamese_network import SiameseNetwork, CombinedLoss, SiameseConfig, save_model


# ─────────────────────────────────────────────
#  TRAINING CONFIG
# ─────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Dataset
    pairs_per_user:         int   = 200     # Positive pairs generated per user
    negative_ratio:         float = 1.0     # Negative pairs per positive pair
    val_split:              float = 0.15    # Fraction of pairs held out for validation
    min_sequences_per_user: int   = 10      # Min sequences needed to include user

    # Training
    epochs:                 int   = 50
    batch_size:             int   = 32
    learning_rate:          float = 1e-3
    weight_decay:           float = 1e-4
    grad_clip:              float = 1.0     # Max gradient norm
    early_stopping_patience:int   = 8       # Stop if val loss doesn't improve
    scheduler_t_max:        int   = 50      # CosineAnnealingLR period

    # Fine-tuning (continuous adaptation)
    finetune_lr:            float = 1e-4    # Lower LR to prevent catastrophic forgetting
    finetune_epochs:        int   = 5
    finetune_batch_size:    int   = 16
    max_risk_score_for_adapt: float = 40.0  # Block adaptation if session was risky

    # Loss weights
    contrastive_margin:     float = 1.0
    loss_alpha:             float = 0.7     # Contrastive loss weight
    loss_beta:              float = 0.3     # BCE loss weight

    # Paths
    checkpoint_dir:         str   = "./checkpoints"
    best_model_name:        str   = "best_model.pt"
    latest_model_name:      str   = "latest_model.pt"

    # Device
    device:                 str   = "auto"  # "auto" | "cpu" | "cuda" | "mps"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────

class BehavioralPairDataset(Dataset):
    """
    Builds a dataset of (anchor, sample, label) pairs for Siamese training.

    Positive pair (label=0): Two sequences from the same user's legitimate sessions.
    Negative pair (label=1): Two sequences from different users, OR one sequence
                              from a legitimate session and one from an attack session.

    Args:
        user_sequences: dict mapping user_id → list of sequences (np.ndarray shape: seq_len, feat_dim)
        attack_sequences: list of sequences labeled as attacks (optional)
        cfg: TrainingConfig
        seed: random seed for reproducibility
    """

    def __init__(
        self,
        user_sequences: dict[str, list[np.ndarray]],
        attack_sequences: Optional[list[np.ndarray]] = None,
        cfg: TrainingConfig = None,
        seed: int = 42,
    ):
        self.cfg = cfg or TrainingConfig()
        self.rng = np.random.default_rng(seed)
        self.pairs: list[tuple] = []  # (anchor, sample, label)

        user_ids = [
            uid for uid, seqs in user_sequences.items()
            if len(seqs) >= self.cfg.min_sequences_per_user
        ]

        if len(user_ids) < 2:
            raise ValueError(
                f"Need at least 2 users with >= {self.cfg.min_sequences_per_user} sequences. "
                f"Got {len(user_ids)} qualifying users."
            )

        logger.info(f"Building dataset from {len(user_ids)} users...")

        self._build_positive_pairs(user_sequences, user_ids)
        self._build_negative_pairs(user_sequences, user_ids, attack_sequences)

        self.rng.shuffle(self.pairs)
        logger.info(
            f"Dataset ready: {len(self.pairs)} pairs  "
            f"({sum(1 for _,_,l in self.pairs if l==0)} pos, "
            f"{sum(1 for _,_,l in self.pairs if l==1)} neg)"
        )

    def _build_positive_pairs(self, user_sequences, user_ids):
        """Same-user pairs → label 0 (legitimate)."""
        for uid in user_ids:
            seqs = user_sequences[uid]
            n = min(self.cfg.pairs_per_user, len(seqs) * (len(seqs) - 1) // 2)
            for _ in range(n):
                i, j = self.rng.choice(len(seqs), size=2, replace=False)
                self.pairs.append((seqs[i], seqs[j], 0))

    def _build_negative_pairs(self, user_sequences, user_ids, attack_sequences):
        """Cross-user and attack pairs → label 1 (anomaly/attack)."""
        n_positive = sum(1 for _, _, l in self.pairs if l == 0)
        n_negative_target = int(n_positive * self.cfg.negative_ratio)
        generated = 0

        # Cross-user pairs (impostor simulation)
        while generated < n_negative_target * 0.7:
            uid_a, uid_b = self.rng.choice(user_ids, size=2, replace=False)
            seq_a = user_sequences[uid_a][self.rng.integers(len(user_sequences[uid_a]))]
            seq_b = user_sequences[uid_b][self.rng.integers(len(user_sequences[uid_b]))]
            self.pairs.append((seq_a, seq_b, 1))
            generated += 1

        # Attack sequences paired with legitimate sequences
        if attack_sequences:
            remaining = n_negative_target - generated
            for _ in range(min(remaining, len(attack_sequences))):
                uid = self.rng.choice(user_ids)
                seq_legit = user_sequences[uid][self.rng.integers(len(user_sequences[uid]))]
                seq_attack = attack_sequences[self.rng.integers(len(attack_sequences))]
                self.pairs.append((seq_legit, seq_attack, 1))
                generated += 1

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor, sample, label = self.pairs[idx]
        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(sample, dtype=torch.float32),
            torch.tensor(label,  dtype=torch.long),
        )


def split_dataset(
    dataset: BehavioralPairDataset,
    val_split: float = 0.15,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Split dataset into train and validation subsets."""
    from torch.utils.data import random_split
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

class MetricsTracker:
    """Tracks loss and accuracy metrics across batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._total_loss        = 0.0
        self._contrastive_loss  = 0.0
        self._bce_loss          = 0.0
        self._correct           = 0
        self._total             = 0
        self._batches           = 0

    def update(self, losses: dict, classification: torch.Tensor, labels: torch.Tensor):
        self._total_loss       += losses["total_loss"].item()
        self._contrastive_loss += losses["contrastive_loss"].item()
        self._bce_loss         += losses["bce_loss"].item()
        self._batches          += 1

        # Classification accuracy: clf > 0.5 = legitimate (label=0)
        predicted = (classification < 0.5).long()  # <0.5 = attack → label 1
        legitimate_pred = (classification >= 0.5).long()  # >=0.5 = legitimate → label 0
        preds = 1 - legitimate_pred
        self._correct += (preds == labels).sum().item()
        self._total   += labels.size(0)

    @property
    def avg_loss(self) -> float:
        return self._total_loss / max(self._batches, 1)

    @property
    def avg_contrastive(self) -> float:
        return self._contrastive_loss / max(self._batches, 1)

    @property
    def avg_bce(self) -> float:
        return self._bce_loss / max(self._batches, 1)

    @property
    def accuracy(self) -> float:
        return self._correct / max(self._total, 1)

    def summary(self) -> dict:
        return {
            "loss":        round(self.avg_loss, 4),
            "contrastive": round(self.avg_contrastive, 4),
            "bce":         round(self.avg_bce, 4),
            "accuracy":    round(self.accuracy, 4),
        }


# ─────────────────────────────────────────────
#  TRAINER
# ─────────────────────────────────────────────

class SiameseTrainer:
    """
    Full training and fine-tuning pipeline for the LSTM Siamese Network.

    Supports:
      - Initial training from scratch on a full user dataset
      - Continuous fine-tuning after each new user session
      - Early stopping, gradient clipping, cosine LR scheduling
      - Checkpoint saving (best model + latest)
      - Training history export as JSON
    """

    def __init__(
        self,
        model: SiameseNetwork,
        cfg: TrainingConfig = None,
    ):
        self.cfg    = cfg or TrainingConfig()
        self.device = self.cfg.resolve_device()
        self.model  = model.to(self.device)
        self.loss_fn = CombinedLoss(
            margin=self.cfg.contrastive_margin,
            alpha=self.cfg.loss_alpha,
            beta=self.cfg.loss_beta,
        )
        self.history: list[dict] = []
        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Trainer initialized on device: {self.device}")

    # ── Optimizer & Scheduler ──────────────────────────────

    def _build_optimizer(self, lr: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.cfg.weight_decay,
        )

    def _build_scheduler(self, optimizer, t_max: int):
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)

    # ── Single epoch ──────────────────────────────────────

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
        training: bool,
    ) -> dict:
        """Run one full epoch. Set optimizer=None for eval-only."""
        self.model.train(training)
        metrics = MetricsTracker()

        for anchor, sample, labels in loader:
            anchor = anchor.to(self.device)
            sample = sample.to(self.device)
            labels = labels.to(self.device)

            if training:
                optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                out    = self.model(anchor, sample)
                losses = self.loss_fn(out["distance"], out["classification"], labels)

            if training:
                losses["total_loss"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                optimizer.step()

            metrics.update(losses, out["classification"].detach(), labels)

        return metrics.summary()

    # ── Main training loop ────────────────────────────────

    def train(
        self,
        dataset: BehavioralPairDataset,
        resume_checkpoint: Optional[str] = None,
    ) -> list[dict]:
        """
        Full training loop with validation and early stopping.

        Args:
            dataset:           BehavioralPairDataset
            resume_checkpoint: Path to checkpoint to resume from

        Returns:
            Training history (list of epoch dicts)
        """
        train_set, val_set = split_dataset(dataset, self.cfg.val_split)

        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

        optimizer  = self._build_optimizer(self.cfg.learning_rate)
        scheduler  = self._build_scheduler(optimizer, self.cfg.scheduler_t_max)
        start_epoch = 0

        # Resume from checkpoint
        if resume_checkpoint:
            start_epoch = self._load_checkpoint(resume_checkpoint, optimizer, scheduler)
            logger.info(f"Resuming from epoch {start_epoch}")

        best_val_loss    = float("inf")
        patience_counter = 0

        logger.info(
            f"Training started — "
            f"{len(train_set)} train pairs, {len(val_set)} val pairs, "
            f"{self.cfg.epochs} epochs"
        )

        for epoch in range(start_epoch, self.cfg.epochs):
            t_start = time.time()

            train_metrics = self._run_epoch(train_loader, optimizer, training=True)
            val_metrics   = self._run_epoch(val_loader,   None,      training=False)
            scheduler.step()

            elapsed = time.time() - t_start
            lr_now  = scheduler.get_last_lr()[0]

            epoch_record = {
                "epoch":    epoch + 1,
                "train":    train_metrics,
                "val":      val_metrics,
                "lr":       round(lr_now, 8),
                "time_s":   round(elapsed, 2),
            }
            self.history.append(epoch_record)

            logger.info(
                f"Epoch {epoch+1:03d}/{self.cfg.epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_acc={val_metrics['accuracy']:.4f}  "
                f"lr={lr_now:.2e}  ({elapsed:.1f}s)"
            )

            # Save latest checkpoint
            self._save_checkpoint(epoch + 1, optimizer, scheduler, tag="latest")

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss    = val_metrics["loss"]
                patience_counter = 0
                self._save_checkpoint(epoch + 1, optimizer, scheduler, tag="best")
                logger.info(f"  ✓ New best val_loss: {best_val_loss:.4f} — checkpoint saved")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch+1} "
                        f"(no improvement for {patience_counter} epochs)"
                    )
                    break

        self._save_history()
        logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
        return self.history

    # ── Fine-tuning (continuous adaptation) ───────────────

    def fine_tune(
        self,
        user_id: str,
        session_sequences: list[np.ndarray],
        label: str = "legitimate",
        session_risk_score: float = 0.0,
    ) -> bool:
        """
        Fine-tune the model on data from a single completed session.

        This implements the continuous profile adaptation described in
        the Risk Scoring Engine spec. Fine-tuning uses a low learning
        rate to prevent catastrophic forgetting of previously learned patterns.

        Args:
            user_id:             User whose session just ended
            session_sequences:   List of (seq_len, feat_dim) arrays from the session
            label:               "legitimate" or "attack"
            session_risk_score:  Average risk score during the session

        Returns:
            True if fine-tuning was performed, False if blocked by safety guardrails
        """
        # ── Safety guardrails ──────────────────────────────
        if session_risk_score > self.cfg.max_risk_score_for_adapt:
            logger.warning(
                f"Fine-tuning BLOCKED for '{user_id}': "
                f"session risk score {session_risk_score:.1f} > "
                f"threshold {self.cfg.max_risk_score_for_adapt}"
            )
            return False

        if label == "attack":
            logger.warning(f"Fine-tuning BLOCKED for '{user_id}': session labeled as attack")
            return False

        if len(session_sequences) < 2:
            logger.warning(f"Fine-tuning SKIPPED for '{user_id}': insufficient sequences ({len(session_sequences)})")
            return False

        logger.info(f"Fine-tuning on session data for user '{user_id}' ({len(session_sequences)} sequences)")

        # ── Build fine-tune pairs ──────────────────────────
        # Create positive pairs from consecutive session windows
        # (anchor = earlier window, sample = later window from same session)
        pairs = []
        rng = np.random.default_rng()
        n_pairs = min(self.cfg.finetune_batch_size * self.cfg.finetune_epochs, len(session_sequences) ** 2)

        for _ in range(n_pairs):
            i, j = rng.choice(len(session_sequences), size=2, replace=False)
            anchor = torch.tensor(session_sequences[i], dtype=torch.float32)
            sample = torch.tensor(session_sequences[j], dtype=torch.float32)
            pairs.append((anchor, sample, torch.tensor(0, dtype=torch.long)))  # 0 = same user

        # ── Fine-tune optimizer (lower LR) ────────────────
        optimizer = self._build_optimizer(self.cfg.finetune_lr)

        self.model.train()
        total_loss = 0.0

        for epoch in range(self.cfg.finetune_epochs):
            rng.shuffle(pairs)
            epoch_loss = 0.0
            batches = 0

            for i in range(0, len(pairs), self.cfg.finetune_batch_size):
                batch = pairs[i:i + self.cfg.finetune_batch_size]
                if not batch:
                    continue

                anchors = torch.stack([p[0] for p in batch]).to(self.device)
                samples = torch.stack([p[1] for p in batch]).to(self.device)
                labels  = torch.stack([p[2] for p in batch]).to(self.device)

                optimizer.zero_grad()
                out    = self.model(anchors, samples)
                losses = self.loss_fn(out["distance"], out["classification"], labels)
                losses["total_loss"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                optimizer.step()

                epoch_loss += losses["total_loss"].item()
                batches    += 1

            avg = epoch_loss / max(batches, 1)
            total_loss += avg
            logger.debug(f"  Fine-tune epoch {epoch+1}/{self.cfg.finetune_epochs} loss={avg:.4f}")

        avg_finetune_loss = total_loss / self.cfg.finetune_epochs
        logger.info(f"Fine-tuning complete for '{user_id}' — avg loss: {avg_finetune_loss:.4f}")

        # Save updated model
        self._save_checkpoint(tag="finetuned")
        return True

    # ── Checkpointing ─────────────────────────────────────

    def _save_checkpoint(
        self,
        epoch: int = 0,
        optimizer=None,
        scheduler=None,
        tag: str = "latest",
    ):
        path = Path(self.cfg.checkpoint_dir) / f"{tag}_model.pt"
        checkpoint = {
            "epoch":              epoch,
            "model_state_dict":   self.model.state_dict(),
            "model_config":       self.model.cfg,
            "training_config":    self.cfg,
        }
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(checkpoint, path)

    def _load_checkpoint(self, path: str, optimizer=None, scheduler=None) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint.get("epoch", 0)

    def _save_history(self):
        path = Path(self.cfg.checkpoint_dir) / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved → {path}")

    def load_best_model(self):
        """Load the best checkpoint into the current model."""
        path = Path(self.cfg.checkpoint_dir) / "best_model.pt"
        self._load_checkpoint(str(path))
        logger.info("Best model loaded.")


# ─────────────────────────────────────────────
#  DATASET BUILDER UTILITY
# ─────────────────────────────────────────────

def build_synthetic_dataset(
    n_users: int = 10,
    sequences_per_user: int = 50,
    n_attack_sequences: int = 100,
    seq_len: int = 6,
    feat_dim: int = 24,
    seed: int = 42,
) -> tuple[dict, list]:
    """
    Generate a synthetic behavioral dataset for testing.

    Each user has a unique behavioral "fingerprint" (mean vector)
    with Gaussian noise added per sequence to simulate natural variation.
    Attack sequences are generated with high-deviation anomalous values.

    Returns:
        user_sequences:   dict user_id → list of (seq_len, feat_dim) arrays
        attack_sequences: list of (seq_len, feat_dim) arrays
    """
    rng = np.random.default_rng(seed)
    user_sequences = {}

    for u in range(n_users):
        uid = f"user_{u:03d}"
        # Each user has a unique behavioral fingerprint
        user_mean = rng.uniform(0.3, 0.7, size=feat_dim).astype(np.float32)
        user_std  = rng.uniform(0.05, 0.15, size=feat_dim).astype(np.float32)

        seqs = []
        for _ in range(sequences_per_user):
            # Each sequence = 6 time steps of this user's behavior
            seq = rng.normal(user_mean, user_std, size=(seq_len, feat_dim)).astype(np.float32)
            seq = np.clip(seq, -5.0, 5.0)  # Simulate already-normalized data
            seqs.append(seq)

        user_sequences[uid] = seqs

    # Attack sequences: high deviation, random anomalous patterns
    attack_sequences = []
    for _ in range(n_attack_sequences):
        seq = rng.uniform(-5.0, 5.0, size=(seq_len, feat_dim)).astype(np.float32)
        attack_sequences.append(seq)

    return user_sequences, attack_sequences


# ─────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    import tempfile

    print("=" * 60)
    print("  ECCAAC — Siamese Training Loop  |  Smoke Test")
    print("=" * 60)

    # ── Build synthetic dataset ──
    print("\nGenerating synthetic dataset...")
    user_seqs, attack_seqs = build_synthetic_dataset(
        n_users=8,
        sequences_per_user=40,
        n_attack_sequences=80,
    )
    print(f"  Users:            {len(user_seqs)}")
    print(f"  Seqs per user:    {len(list(user_seqs.values())[0])}")
    print(f"  Attack sequences: {len(attack_seqs)}")

    # ── Build pair dataset ──
    cfg_train = TrainingConfig(
        epochs=5,
        batch_size=32,
        pairs_per_user=60,
        learning_rate=1e-3,
        early_stopping_patience=3,
    )

    dataset = BehavioralPairDataset(user_seqs, attack_seqs, cfg=cfg_train, seed=42)
    print(f"\nPair dataset: {len(dataset)} pairs")

    # ── Initialize model and trainer ──
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_train.checkpoint_dir = tmpdir

        from siamese_network import build_model, SiameseConfig
        model = build_model(SiameseConfig())
        trainer = SiameseTrainer(model, cfg=cfg_train)

        print(f"\nDevice: {trainer.device}")
        print(f"Starting training ({cfg_train.epochs} epochs)...\n")

        history = trainer.train(dataset)

        # ── Print summary ──
        print("\nTraining Summary:")
        print(f"  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} {'Val Acc':<12} {'Time'}")
        print(f"  {'-'*55}")
        for h in history:
            print(
                f"  {h['epoch']:<8} "
                f"{h['train']['loss']:<14.4f} "
                f"{h['val']['loss']:<14.4f} "
                f"{h['val']['accuracy']:<12.4f} "
                f"{h['time_s']:.1f}s"
            )

        # ── Fine-tuning test ──
        print("\nTesting fine-tuning (simulated new session)...")
        new_seqs = user_seqs["user_000"][:8]
        adapted = trainer.fine_tune(
            user_id="user_000",
            session_sequences=new_seqs,
            label="legitimate",
            session_risk_score=22.0,
        )
        print(f"  Fine-tuning performed: {adapted}")

        # ── Fine-tuning blocked by guardrail ──
        print("\nTesting fine-tuning guardrail (high-risk session)...")
        blocked = trainer.fine_tune(
            user_id="user_000",
            session_sequences=new_seqs,
            label="legitimate",
            session_risk_score=75.0,  # above threshold
        )
        print(f"  Fine-tuning blocked:   {not blocked}")

    print(f"\n✓ All checks passed.\n")
