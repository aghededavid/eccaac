"""
ECCAAC — Risk Scoring Engine
Feature Preprocessor & Normalizer

Responsibilities:
  - Accept raw behavioral signals from the Go agent
  - Validate and clean incoming feature values
  - Apply per-user Z-score normalization (running stats)
  - Maintain and persist per-user normalization parameters
  - Assemble normalized vectors into time sequences
  - Handle cold-start users with global fallback stats

Feature Vector (24 dimensions):
  Typing Dynamics  [0:6]   — hold_time_mean, hold_time_std, latency_mean,
                              latency_std, typing_speed, error_rate
  Mouse Dynamics   [6:11]  — mouse_speed, acceleration, click_interval,
                              movement_entropy, idle_mouse_ratio
  Device Context   [11:14] — cpu_load, memory_usage, device_hash_embedding
  Process Activity [14:17] — process_count, suspicious_flag, foreground_app
  Network Context  [17:21] — ip_change_flag, geo_distance, packet_rate, vpn_flag
  Session Behavior [21:24] — session_duration, idle_ratio, transition_rate
"""

import json
import math
import time
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

FEATURE_DIM = 24
SEQUENCE_LEN = 6          # 6 vectors = 60 seconds of behavior
WINDOW_SECONDS = 10       # Each vector covers a 10-second window
MIN_SAMPLES_FOR_USER_STATS = 30   # Minimum samples before using user stats
WARMUP_SESSIONS = 5       # Cold-start observation period


# Feature index map — single source of truth
FEATURE_INDEX = {
    # Typing Dynamics
    "hold_time_mean":       0,
    "hold_time_std":        1,
    "latency_mean":         2,
    "latency_std":          3,
    "typing_speed":         4,
    "error_rate":           5,
    # Mouse Dynamics
    "mouse_speed":          6,
    "mouse_acceleration":   7,
    "click_interval":       8,
    "movement_entropy":     9,
    "idle_mouse_ratio":     10,
    # Device Context
    "cpu_load":             11,
    "memory_usage":         12,
    "device_hash_embed":    13,
    # Process Activity
    "process_count":        14,
    "suspicious_flag":      15,
    "foreground_app":       16,
    # Network Context
    "ip_change_flag":       17,
    "geo_distance":         18,
    "packet_rate":          19,
    "vpn_flag":             20,
    # Session Behavior
    "session_duration":     21,
    "idle_ratio":           22,
    "transition_rate":      23,
}

# Feature value bounds for validation — (min, max)
FEATURE_BOUNDS = {
    "hold_time_mean":       (0.0,   2000.0),   # ms
    "hold_time_std":        (0.0,   500.0),
    "latency_mean":         (0.0,   5000.0),   # ms
    "latency_std":          (0.0,   1000.0),
    "typing_speed":         (0.0,   25.0),     # chars/sec
    "error_rate":           (0.0,   1.0),      # ratio
    "mouse_speed":          (0.0,   5000.0),   # px/sec
    "mouse_acceleration":   (0.0,   2000.0),   # px/sec²
    "click_interval":       (0.0,   30.0),     # seconds
    "movement_entropy":     (0.0,   1.0),
    "idle_mouse_ratio":     (0.0,   1.0),
    "cpu_load":             (0.0,   1.0),      # ratio
    "memory_usage":         (0.0,   1.0),
    "device_hash_embed":    (0.0,   1.0),      # normalized embedding
    "process_count":        (0.0,   1000.0),
    "suspicious_flag":      (0.0,   1.0),      # binary
    "foreground_app":       (0.0,   50.0),     # encoded category
    "ip_change_flag":       (0.0,   1.0),      # binary
    "geo_distance":         (0.0,   20000.0),  # km
    "packet_rate":          (0.0,   100000.0), # packets/sec
    "vpn_flag":             (0.0,   1.0),      # binary
    "session_duration":     (0.0,   86400.0),  # seconds (max 24h)
    "idle_ratio":           (0.0,   1.0),
    "transition_rate":      (0.0,   100.0),    # transitions/min
}

# Global fallback stats (population-level defaults)
# Used for cold-start users before personal stats are established
GLOBAL_MEAN = np.array([
    120.0, 30.0, 95.0, 20.0, 4.8, 0.03,   # Typing
    850.0, 120.0, 0.9, 0.63, 0.18,         # Mouse
    0.45, 0.55, 0.5,                        # Device
    85.0, 0.05, 1.0,                        # Process
    0.1, 10.0, 300.0, 0.2,                 # Network
    300.0, 0.25, 3.5,                       # Session
], dtype=np.float32)

GLOBAL_STD = np.array([
    40.0, 15.0, 35.0, 10.0, 1.5, 0.02,    # Typing
    300.0, 60.0, 0.5, 0.2, 0.1,            # Mouse
    0.25, 0.2, 0.3,                         # Device
    40.0, 0.1, 0.5,                         # Process
    0.3, 30.0, 200.0, 0.4,                 # Network
    180.0, 0.15, 2.0,                       # Session
], dtype=np.float32)


# ─────────────────────────────────────────────
#  RAW FEATURE INPUT
# ─────────────────────────────────────────────

@dataclass
class RawFeatures:
    """
    Raw behavioral signals sent from the Go agent.
    All values are in their natural units before normalization.
    Missing values default to -1.0 (treated as unavailable).
    """
    # Typing Dynamics
    hold_time_mean:     float = -1.0
    hold_time_std:      float = -1.0
    latency_mean:       float = -1.0
    latency_std:        float = -1.0
    typing_speed:       float = -1.0
    error_rate:         float = -1.0
    # Mouse Dynamics
    mouse_speed:        float = -1.0
    mouse_acceleration: float = -1.0
    click_interval:     float = -1.0
    movement_entropy:   float = -1.0
    idle_mouse_ratio:   float = -1.0
    # Device Context
    cpu_load:           float = -1.0
    memory_usage:       float = -1.0
    device_hash_embed:  float = -1.0
    # Process Activity
    process_count:      float = -1.0
    suspicious_flag:    float = -1.0
    foreground_app:     float = -1.0
    # Network Context
    ip_change_flag:     float = -1.0
    geo_distance:       float = -1.0
    packet_rate:        float = -1.0
    vpn_flag:           float = -1.0
    # Session Behavior
    session_duration:   float = -1.0
    idle_ratio:         float = -1.0
    transition_rate:    float = -1.0
    # Metadata
    user_id:            str   = ""
    device_id:          str   = ""
    session_id:         str   = ""
    timestamp:          float = field(default_factory=time.time)

    def to_array(self) -> np.ndarray:
        """Convert to ordered numpy array matching FEATURE_INDEX."""
        return np.array([
            self.hold_time_mean, self.hold_time_std,
            self.latency_mean, self.latency_std,
            self.typing_speed, self.error_rate,
            self.mouse_speed, self.mouse_acceleration,
            self.click_interval, self.movement_entropy,
            self.idle_mouse_ratio,
            self.cpu_load, self.memory_usage, self.device_hash_embed,
            self.process_count, self.suspicious_flag, self.foreground_app,
            self.ip_change_flag, self.geo_distance,
            self.packet_rate, self.vpn_flag,
            self.session_duration, self.idle_ratio, self.transition_rate,
        ], dtype=np.float32)

    @classmethod
    def from_dict(cls, d: dict) -> "RawFeatures":
        """Construct from a dictionary (e.g. deserialized gRPC payload)."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# ─────────────────────────────────────────────
#  PER-USER RUNNING STATISTICS
# ─────────────────────────────────────────────

@dataclass
class UserStats:
    """
    Welford's online algorithm for per-user running mean and variance.
    Updates incrementally — no need to store raw historical data.

    Reference: Welford (1962), "Note on a Method for Calculating
    Corrected Sums of Squares and Products"
    """
    user_id:    str
    n:          int         = 0            # total samples seen
    mean:       np.ndarray  = None         # running mean
    M2:         np.ndarray  = None         # running sum of squared deviations
    sessions:   int         = 0           # completed sessions
    created_at: float       = field(default_factory=time.time)
    updated_at: float       = field(default_factory=time.time)

    def __post_init__(self):
        if self.mean is None:
            self.mean = GLOBAL_MEAN.copy()
        if self.M2 is None:
            self.M2 = (GLOBAL_STD ** 2).copy()

    @property
    def std(self) -> np.ndarray:
        """Population standard deviation with floor to prevent div-by-zero."""
        if self.n < 2:
            return GLOBAL_STD.copy()
        variance = self.M2 / (self.n - 1)
        return np.sqrt(np.maximum(variance, 1e-8))

    @property
    def is_mature(self) -> bool:
        """True when enough samples exist to rely on personal stats."""
        return self.n >= MIN_SAMPLES_FOR_USER_STATS

    def update(self, x: np.ndarray):
        """
        Update running stats with one new feature vector.
        Uses Welford's online algorithm for numerical stability.
        """
        # Only update on valid (non-missing) features
        valid_mask = x >= 0
        if not valid_mask.any():
            return

        self.n += 1
        delta = np.where(valid_mask, x - self.mean, 0.0)
        self.mean = np.where(valid_mask, self.mean + delta / self.n, self.mean)
        delta2 = np.where(valid_mask, x - self.mean, 0.0)
        self.M2 = np.where(valid_mask, self.M2 + delta * delta2, self.M2)
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "user_id":    self.user_id,
            "n":          self.n,
            "sessions":   self.sessions,
            "mean":       self.mean.tolist(),
            "M2":         self.M2.tolist(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserStats":
        obj = cls(user_id=d["user_id"])
        obj.n          = d["n"]
        obj.sessions   = d.get("sessions", 0)
        obj.mean       = np.array(d["mean"], dtype=np.float32)
        obj.M2         = np.array(d["M2"], dtype=np.float32)
        obj.created_at = d.get("created_at", time.time())
        obj.updated_at = d.get("updated_at", time.time())
        return obj


# ─────────────────────────────────────────────
#  FEATURE VALIDATOR
# ─────────────────────────────────────────────

class FeatureValidator:
    """
    Validates and cleans a raw feature array before normalization.

    Rules:
      - Values of -1.0 are treated as missing → imputed with user mean
      - Values outside FEATURE_BOUNDS are clipped with a warning
      - NaN / Inf values are replaced with user mean
    """

    def validate_and_clean(
        self,
        raw: np.ndarray,
        user_mean: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        """
        Args:
            raw:       Raw 24-dim feature array
            user_mean: Per-user mean for imputation

        Returns:
            (cleaned_array, list_of_warnings)
        """
        assert raw.shape == (FEATURE_DIM,), f"Expected shape ({FEATURE_DIM},), got {raw.shape}"
        cleaned = raw.copy()
        warnings = []
        feature_names = list(FEATURE_INDEX.keys())

        for i, name in enumerate(feature_names):
            val = cleaned[i]

            # Missing value (-1.0) → impute with user mean
            if val < 0:
                cleaned[i] = user_mean[i]
                warnings.append(f"Imputed missing value for '{name}'")
                continue

            # NaN / Inf → impute with user mean
            if not math.isfinite(val):
                cleaned[i] = user_mean[i]
                warnings.append(f"Replaced NaN/Inf for '{name}'")
                continue

            # Clip to bounds
            lo, hi = FEATURE_BOUNDS[name]
            if val < lo or val > hi:
                cleaned[i] = np.clip(val, lo, hi)
                warnings.append(f"Clipped '{name}': {val:.2f} → [{lo}, {hi}]")

        return cleaned, warnings


# ─────────────────────────────────────────────
#  NORMALIZER
# ─────────────────────────────────────────────

class FeatureNormalizer:
    """
    Per-user Z-score normalization.

    For mature users (n >= 30), uses personal mean/std.
    For cold-start users, blends personal and global stats.

    Formula: x_norm = (x - mean) / std
    """

    def normalize(
        self,
        x: np.ndarray,
        stats: UserStats
    ) -> np.ndarray:
        """
        Normalize a cleaned feature vector using user stats.

        Args:
            x:     Cleaned 24-dim feature array
            stats: Per-user running statistics

        Returns:
            Normalized array, clipped to [-5, 5] to prevent extreme outliers
        """
        if stats.is_mature:
            mean = stats.mean
            std  = stats.std
        else:
            # Blend user stats with global fallback
            # As n grows toward MIN_SAMPLES, trust shifts to personal stats
            alpha = stats.n / MIN_SAMPLES_FOR_USER_STATS
            mean = alpha * stats.mean + (1 - alpha) * GLOBAL_MEAN
            std  = alpha * stats.std  + (1 - alpha) * GLOBAL_STD

        normalized = (x - mean) / np.maximum(std, 1e-8)

        # Clip extreme outliers — beyond 5σ is almost certainly noise or attack
        normalized = np.clip(normalized, -5.0, 5.0)

        return normalized.astype(np.float32)

    def denormalize(
        self,
        x_norm: np.ndarray,
        stats: UserStats
    ) -> np.ndarray:
        """Reverse normalization — for debugging and logging."""
        return (x_norm * stats.std) + stats.mean


# ─────────────────────────────────────────────
#  SEQUENCE BUFFER
# ─────────────────────────────────────────────

class SequenceBuffer:
    """
    Accumulates normalized feature vectors into fixed-length sequences.

    Maintains a sliding window of the last SEQUENCE_LEN vectors per session.
    Once full, yields a (SEQUENCE_LEN, FEATURE_DIM) numpy array ready for
    the Siamese network.
    """

    def __init__(self, seq_len: int = SEQUENCE_LEN):
        self.seq_len = seq_len
        self._buffer: deque = deque(maxlen=seq_len)

    def push(self, vector: np.ndarray):
        """Add one normalized vector to the buffer."""
        assert vector.shape == (FEATURE_DIM,)
        self._buffer.append(vector.copy())

    @property
    def is_ready(self) -> bool:
        """True when the buffer holds a full sequence."""
        return len(self._buffer) == self.seq_len

    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Returns the current sequence as (seq_len, feature_dim) array,
        or None if buffer is not yet full.
        """
        if not self.is_ready:
            return None
        return np.stack(list(self._buffer), axis=0)  # (seq_len, feature_dim)

    def reset(self):
        """Clear the buffer — call at session start."""
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)


# ─────────────────────────────────────────────
#  STATS STORE  (file-based persistence)
# ─────────────────────────────────────────────

class UserStatsStore:
    """
    Persists per-user normalization statistics to disk.
    In production this would be backed by PostgreSQL.
    For now, uses JSON files — one per user.
    """

    def __init__(self, storage_dir: str = "./user_stats"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, UserStats] = {}

    def _path(self, user_id: str) -> Path:
        safe_id = user_id.replace("/", "_").replace("\\", "_")
        return self.storage_dir / f"{safe_id}.json"

    def load(self, user_id: str) -> UserStats:
        """Load stats for a user. Creates new stats if not found."""
        if user_id in self._cache:
            return self._cache[user_id]

        path = self._path(user_id)
        if path.exists():
            with open(path) as f:
                stats = UserStats.from_dict(json.load(f))
            logger.debug(f"Loaded stats for user '{user_id}' (n={stats.n})")
        else:
            stats = UserStats(user_id=user_id)
            logger.info(f"New user '{user_id}' — initialized default stats")

        self._cache[user_id] = stats
        return stats

    def save(self, stats: UserStats):
        """Persist stats to disk and update cache."""
        self._cache[stats.user_id] = stats
        path = self._path(stats.user_id)
        with open(path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)
        logger.debug(f"Saved stats for user '{stats.user_id}' (n={stats.n})")

    def exists(self, user_id: str) -> bool:
        return self._path(user_id).exists() or user_id in self._cache

    def get_all_user_ids(self) -> list[str]:
        return [p.stem for p in self.storage_dir.glob("*.json")]


# ─────────────────────────────────────────────
#  MAIN PREPROCESSOR
# ─────────────────────────────────────────────

class FeaturePreprocessor:
    """
    Main preprocessing pipeline for the ECCAAC Risk Scoring Engine.

    For each incoming RawFeatures object, this pipeline:
      1. Validates and cleans the raw array
      2. Updates per-user running statistics
      3. Normalizes the vector using per-user Z-score stats
      4. Pushes the normalized vector into the session sequence buffer
      5. Returns the current sequence if the buffer is full

    One FeaturePreprocessor instance is shared across all sessions.
    Each session gets its own SequenceBuffer.
    """

    def __init__(self, storage_dir: str = "./user_stats"):
        self.validator  = FeatureValidator()
        self.normalizer = FeatureNormalizer()
        self.stats_store = UserStatsStore(storage_dir)
        self._session_buffers: dict[str, SequenceBuffer] = {}

    def _get_buffer(self, session_id: str) -> SequenceBuffer:
        if session_id not in self._session_buffers:
            self._session_buffers[session_id] = SequenceBuffer()
        return self._session_buffers[session_id]

    def process(
        self,
        raw: RawFeatures
    ) -> dict:
        """
        Process one raw feature sample through the full pipeline.

        Args:
            raw: RawFeatures from the Go agent

        Returns:
            dict with:
              normalized_vector:  np.ndarray (24,) — always present
              sequence:           np.ndarray (6, 24) or None if buffer not full
              sequence_ready:     bool
              user_stats_mature:  bool
              warnings:           list[str]
              user_session_count: int
        """
        # 1. Load user stats
        stats = self.stats_store.load(raw.user_id)

        # 2. Convert to array and validate
        raw_array = raw.to_array()
        cleaned, warnings = self.validator.validate_and_clean(raw_array, stats.mean)

        # 3. Update running stats with cleaned values
        stats.update(cleaned)

        # 4. Normalize
        normalized = self.normalizer.normalize(cleaned, stats)

        # 5. Push into session sequence buffer
        buffer = self._get_buffer(raw.session_id)
        buffer.push(normalized)

        # 6. Get sequence if ready
        sequence = buffer.get_sequence()

        # Persist updated stats (in production: batch writes)
        self.stats_store.save(stats)

        if warnings:
            logger.debug(f"Validation warnings for {raw.user_id}: {warnings}")

        return {
            "normalized_vector":  normalized,
            "sequence":           sequence,
            "sequence_ready":     sequence is not None,
            "user_stats_mature":  stats.is_mature,
            "user_session_count": stats.sessions,
            "warnings":           warnings,
        }

    def on_session_end(self, user_id: str, session_id: str):
        """
        Called when a session ends cleanly.
        Increments session counter and cleans up buffer.
        """
        stats = self.stats_store.load(user_id)
        stats.sessions += 1
        self.stats_store.save(stats)

        if session_id in self._session_buffers:
            del self._session_buffers[session_id]

        logger.info(f"Session ended for user '{user_id}' — total sessions: {stats.sessions}")

    def get_user_phase(self, user_id: str) -> str:
        """
        Returns the current onboarding phase for a user.
          OBSERVATION    — first 5 sessions, no enforcement
          SOFT           — sessions 6–10, relaxed thresholds
          FULL           — session 11+, full enforcement
        """
        stats = self.stats_store.load(user_id)
        if stats.sessions < WARMUP_SESSIONS:
            return "OBSERVATION"
        elif stats.sessions < WARMUP_SESSIONS * 2:
            return "SOFT"
        else:
            return "FULL"

    def reset_session_buffer(self, session_id: str):
        """Manually reset a session buffer (e.g. after step-up auth)."""
        if session_id in self._session_buffers:
            self._session_buffers[session_id].reset()


# ─────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("=" * 55)
    print("  ECCAAC — Feature Preprocessor  |  Smoke Test")
    print("=" * 55)

    with tempfile.TemporaryDirectory() as tmpdir:
        preprocessor = FeaturePreprocessor(storage_dir=tmpdir)

        user_id    = "user_david_001"
        session_id = "session_abc_123"

        print(f"\nUser:    {user_id}")
        print(f"Session: {session_id}")
        print(f"Phase:   {preprocessor.get_user_phase(user_id)}")

        # Simulate 8 windows of raw signals (80 seconds)
        print(f"\nProcessing 8 raw feature windows...")
        for i in range(8):
            raw = RawFeatures(
                # Typing
                hold_time_mean=120.0 + np.random.normal(0, 5),
                hold_time_std=30.0,
                latency_mean=95.0 + np.random.normal(0, 3),
                latency_std=20.0,
                typing_speed=4.8,
                error_rate=0.03,
                # Mouse
                mouse_speed=850.0,
                mouse_acceleration=120.0,
                click_interval=0.9,
                movement_entropy=0.63,
                idle_mouse_ratio=0.18,
                # Device
                cpu_load=0.45,
                memory_usage=0.61,
                device_hash_embed=0.72,
                # Process
                process_count=85,
                suspicious_flag=0.0,
                foreground_app=1.0,
                # Network
                ip_change_flag=0.0,
                geo_distance=0.0,
                packet_rate=200.0,
                vpn_flag=0.0,
                # Session
                session_duration=float(60 + i * 10),
                idle_ratio=0.25,
                transition_rate=3.2,
                # Metadata
                user_id=user_id,
                device_id="device_001",
                session_id=session_id,
            )

            result = preprocessor.process(raw)
            status = "SEQUENCE READY ✓" if result["sequence_ready"] else f"buffer {i+1}/6"
            print(f"  Window {i+1:02d}: {status}  |  warnings: {len(result['warnings'])}")

            if result["sequence_ready"]:
                seq = result["sequence"]
                print(f"           Sequence shape: {seq.shape}")
                print(f"           Mean norm val:  {seq.mean():.4f}")
                print(f"           Std  norm val:  {seq.std():.4f}")

        # Test missing value imputation
        print(f"\nTesting missing value imputation...")
        raw_missing = RawFeatures(
            hold_time_mean=-1.0,   # missing
            latency_mean=95.0,
            typing_speed=-1.0,     # missing
            cpu_load=0.5,
            memory_usage=0.6,
            user_id=user_id,
            session_id=session_id,
        )
        result = preprocessor.process(raw_missing)
        print(f"  Warnings: {result['warnings'][:3]}")

        # Test anomalous values (attack simulation)
        print(f"\nTesting anomalous input (simulated attack)...")
        raw_attack = RawFeatures(
            hold_time_mean=800.0,   # very abnormal
            typing_speed=0.1,       # very slow
            ip_change_flag=1.0,
            geo_distance=5000.0,    # 5000km away
            suspicious_flag=1.0,
            process_count=400.0,
            user_id=user_id,
            session_id="session_attack_999",
        )
        result_attack = preprocessor.process(raw_attack)
        norm_vec = result_attack["normalized_vector"]
        print(f"  hold_time_mean normalized:  {norm_vec[0]:.3f}σ")
        print(f"  geo_distance normalized:    {norm_vec[18]:.3f}σ")
        print(f"  typing_speed normalized:    {norm_vec[4]:.3f}σ")

        # Session end
        preprocessor.on_session_end(user_id, session_id)
        print(f"\nPhase after session end: {preprocessor.get_user_phase(user_id)}")

    print(f"\n✓ All checks passed.\n")
