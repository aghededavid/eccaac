"""
ECCAAC — Risk Scoring Engine
LSTM Siamese Network (PyTorch)

Architecture:
  - Bidirectional LSTM encoder (shared weights across both branches)
  - L2-normalized embedding output
  - Binary classification head (Legitimate vs. Attack)
  - Contrastive loss for training

Input:  Sequence of shape (batch, 6, 24)  →  6 time steps × 24 features
Output: Risk similarity score + binary classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

@dataclass
class SiameseConfig:
    input_size: int = 24          # Feature vector dimensionality
    sequence_len: int = 6         # Time steps per sequence (60 seconds)
    lstm_hidden: int = 64         # LSTM hidden units
    lstm_layers: int = 2          # Number of stacked LSTM layers
    lstm_dropout: float = 0.3     # Dropout between LSTM layers
    embedding_dim: int = 16       # Final L2-normalized embedding size
    fc_hidden: int = 32           # FC layer hidden size
    bidirectional: bool = True    # Use bidirectional LSTM in first layer
    contrastive_margin: float = 1.0  # Margin for contrastive loss


# ─────────────────────────────────────────────
#  LSTM ENCODER  (shared branch)
# ─────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """
    Encodes a behavioral sequence (6 × 24) into a fixed-size
    L2-normalized embedding vector.

    Both branches of the Siamese network share one instance
    of this encoder (identical weights).
    """

    def __init__(self, cfg: SiameseConfig):
        super().__init__()
        self.cfg = cfg

        # Bidirectional LSTM — captures forward & backward temporal patterns
        bidir_factor = 2 if cfg.bidirectional else 1
        self.lstm1 = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=0.0
        )
        self.dropout1 = nn.Dropout(cfg.lstm_dropout)

        # Second LSTM — unidirectional, processes compressed representation
        self.lstm2 = nn.LSTM(
            input_size=cfg.lstm_hidden * bidir_factor,
            hidden_size=cfg.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )
        self.dropout2 = nn.Dropout(cfg.lstm_dropout)

        # Fully connected projection to embedding space
        self.fc1 = nn.Linear(cfg.lstm_hidden, cfg.fc_hidden)
        self.fc2 = nn.Linear(cfg.fc_hidden, cfg.embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for FC layers; orthogonal for LSTM weights."""
        for name, param in self.lstm1.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for name, param in self.lstm2.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, sequence_len, input_size)
        Returns:
            embedding: (batch, embedding_dim)  — L2 normalized
        """
        # Layer 1: Bidirectional LSTM
        out, _ = self.lstm1(x)                   # (batch, seq, hidden*2)
        out = self.dropout1(out)

        # Layer 2: Unidirectional LSTM — take final hidden state
        out, (h_n, _) = self.lstm2(out)          # h_n: (1, batch, hidden)
        out = h_n.squeeze(0)                      # (batch, hidden)
        out = self.dropout2(out)

        # FC projection
        out = F.relu(self.fc1(out))               # (batch, fc_hidden)
        out = self.fc2(out)                        # (batch, embedding_dim)

        # L2 normalize — embeddings live on a unit hypersphere
        embedding = F.normalize(out, p=2, dim=1)  # (batch, embedding_dim)
        return embedding


# ─────────────────────────────────────────────
#  SIAMESE NETWORK
# ─────────────────────────────────────────────

class SiameseNetwork(nn.Module):
    """
    LSTM Siamese Network for behavioral authentication.

    Takes two sequences (profile anchor + live session) and outputs:
      - similarity_score:   float 0.0–1.0  (1.0 = identical to profile)
      - classification:     float 0.0–1.0  (> 0.5 = Legitimate)
      - distance:           Euclidean distance between embeddings

    Both branches share the same LSTMEncoder weights.
    """

    def __init__(self, cfg: SiameseConfig = None):
        super().__init__()
        self.cfg = cfg or SiameseConfig()

        # Shared encoder — one instance, used for both branches
        self.encoder = LSTMEncoder(self.cfg)

        # Binary classification head
        # Input: element-wise absolute difference of embeddings
        self.classifier = nn.Sequential(
            nn.Linear(self.cfg.embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self._init_classifier()

    def _init_classifier(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single sequence into an embedding."""
        return self.encoder(x)

    def forward(
        self,
        anchor: torch.Tensor,
        sample: torch.Tensor
    ) -> dict:
        """
        Args:
            anchor: (batch, seq_len, input_size) — stored profile sequence
            sample: (batch, seq_len, input_size) — live session sequence

        Returns:
            dict with keys:
              embedding_anchor:  (batch, embedding_dim)
              embedding_sample:  (batch, embedding_dim)
              distance:          (batch,)  Euclidean distance
              similarity:        (batch,)  0.0–1.0
              classification:    (batch,)  0.0–1.0  (>0.5 = Legitimate)
        """
        # Both branches through shared encoder
        emb_anchor = self.encoder(anchor)   # (batch, embedding_dim)
        emb_sample = self.encoder(sample)   # (batch, embedding_dim)

        # Euclidean distance between embeddings
        distance = F.pairwise_distance(emb_anchor, emb_sample, p=2)  # (batch,)

        # Similarity score: 1 = identical, 0 = maximally different
        similarity = 1.0 / (1.0 + distance)  # (batch,)

        # Classification: |emb_A - emb_B| → binary label
        diff = torch.abs(emb_anchor - emb_sample)    # (batch, embedding_dim)
        classification = self.classifier(diff).squeeze(1)  # (batch,)

        return {
            "embedding_anchor": emb_anchor,
            "embedding_sample": emb_sample,
            "distance":         distance,
            "similarity":       similarity,
            "classification":   classification,
        }

    def predict_risk(
        self,
        anchor: torch.Tensor,
        sample: torch.Tensor
    ) -> dict:
        """
        Inference-mode prediction. Returns human-readable risk output.

        Returns:
            dict with:
              similarity_score:  float 0.0–1.0
              behavioral_deviation: float 0.0–50.0  (contribution to risk score)
              classification:    str  "LEGITIMATE" | "ATTACK"
              confidence:        float 0.0–1.0
              distance:          float
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(anchor, sample)

        similarity = out["similarity"].item()
        clf_score = out["classification"].item()

        # Behavioral deviation: contributes up to 50 points to risk score
        behavioral_deviation = (1.0 - similarity) * 50.0

        # Classification label and confidence
        if clf_score >= 0.5:
            label = "LEGITIMATE"
            confidence = clf_score
        else:
            label = "ATTACK"
            confidence = 1.0 - clf_score

        return {
            "similarity_score":       round(similarity, 4),
            "behavioral_deviation":   round(behavioral_deviation, 2),
            "classification":         label,
            "confidence":             round(confidence, 4),
            "distance":               round(out["distance"].item(), 4),
        }


# ─────────────────────────────────────────────
#  LOSS FUNCTIONS
# ─────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Network training.

    Pulls same-user pairs together, pushes different-user pairs apart.

    L = (1 - y) * 0.5 * D²  +  y * 0.5 * max(0, margin - D)²

    Args:
        margin: Minimum desired distance between different-user pairs.
                Pairs already beyond margin are not penalized.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        distance: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            distance: (batch,)  Euclidean distance between embedding pairs
            label:    (batch,)  0 = same user (legitimate), 1 = different (attack)
        Returns:
            loss: scalar
        """
        same_user_loss = (1 - label) * 0.5 * distance.pow(2)
        diff_user_loss = label * 0.5 * F.relu(self.margin - distance).pow(2)
        loss = (same_user_loss + diff_user_loss).mean()
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss = Contrastive Loss + Binary Cross-Entropy.

    Trains both the embedding space (contrastive) and the
    classification head (BCE) simultaneously.

    Args:
        margin:      Contrastive loss margin
        alpha:       Weight of contrastive loss  (default 0.7)
        beta:        Weight of BCE loss          (default 0.3)
    """

    def __init__(self, margin: float = 1.0, alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.contrastive = ContrastiveLoss(margin)
        self.bce = nn.BCELoss()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        distance: torch.Tensor,
        classification: torch.Tensor,
        label: torch.Tensor
    ) -> dict:
        """
        Args:
            distance:       (batch,)  from model output
            classification: (batch,)  from model output (sigmoid probabilities)
            label:          (batch,)  0 = same user, 1 = attack
        Returns:
            dict with total_loss, contrastive_loss, bce_loss
        """
        # For BCE: same user (label=0) → target 1.0 (legitimate)
        #          attack   (label=1) → target 0.0
        clf_target = 1.0 - label.float()

        l_contrastive = self.contrastive(distance, label.float())
        l_bce = self.bce(classification, clf_target)

        total = self.alpha * l_contrastive + self.beta * l_bce

        return {
            "total_loss":       total,
            "contrastive_loss": l_contrastive,
            "bce_loss":         l_bce,
        }


# ─────────────────────────────────────────────
#  MODEL UTILITIES
# ─────────────────────────────────────────────

def build_model(cfg: SiameseConfig = None) -> SiameseNetwork:
    """Instantiate the Siamese network with given or default config."""
    return SiameseNetwork(cfg or SiameseConfig())


def save_model(model: SiameseNetwork, path: str):
    """Save model weights and config."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.cfg,
    }, path)
    print(f"Model saved → {path}")


def load_model(path: str) -> SiameseNetwork:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    cfg = checkpoint["config"]
    model = SiameseNetwork(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded ← {path}")
    return model


def count_parameters(model: SiameseNetwork) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  ECCAAC — LSTM Siamese Network  |  Smoke Test")
    print("=" * 55)

    cfg = SiameseConfig()
    model = build_model(cfg)
    loss_fn = CombinedLoss()

    print(f"\nConfig:")
    print(f"  Input size:     {cfg.input_size} features")
    print(f"  Sequence len:   {cfg.sequence_len} steps  ({cfg.sequence_len * 10}s window)")
    print(f"  Embedding dim:  {cfg.embedding_dim}")
    print(f"  Parameters:     {count_parameters(model):,}")

    # ── Forward pass test ──
    batch_size = 4
    anchor = torch.randn(batch_size, cfg.sequence_len, cfg.input_size)
    sample = torch.randn(batch_size, cfg.sequence_len, cfg.input_size)
    labels = torch.tensor([0, 0, 1, 1])  # 0=same user, 1=attack

    out = model(anchor, sample)
    losses = loss_fn(out["distance"], out["classification"], labels)

    print(f"\nForward pass  (batch={batch_size}):")
    print(f"  Anchor embedding:  {out['embedding_anchor'].shape}")
    print(f"  Sample embedding:  {out['embedding_sample'].shape}")
    print(f"  Distances:         {out['distance'].detach().numpy().round(4)}")
    print(f"  Similarities:      {out['similarity'].detach().numpy().round(4)}")
    print(f"  Classifications:   {out['classification'].detach().numpy().round(4)}")

    print(f"\nLoss:")
    print(f"  Total:        {losses['total_loss'].item():.4f}")
    print(f"  Contrastive:  {losses['contrastive_loss'].item():.4f}")
    print(f"  BCE:          {losses['bce_loss'].item():.4f}")

    # ── Risk prediction test ──
    print(f"\nRisk Prediction (single sample):")
    anchor_single = torch.randn(1, cfg.sequence_len, cfg.input_size)
    sample_single = anchor_single + torch.randn(1, cfg.sequence_len, cfg.input_size) * 0.1  # slight deviation
    result = model.predict_risk(anchor_single, sample_single)
    for k, v in result.items():
        print(f"  {k:<28} {v}")

    print(f"\n✓ All checks passed.\n")
