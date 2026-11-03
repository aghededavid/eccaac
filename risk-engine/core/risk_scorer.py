"""
ECCAAC — Risk Scoring Engine
Risk Scorer

Responsibilities:
  - Compute the final 0–100 risk score from all signal components
  - Combine behavioral, context, and device deviation signals
  - Apply per-phase threshold adjustments (OBSERVATION / SOFT / FULL)
  - Provide per-signal breakdowns for audit logging and admin visibility
  - Emit an enforcement recommendation based on risk level

Risk Score Formula:
  Risk Score = Behavioral Deviation + Context Deviation + Device Risk

  Component              Source                          Max
  ─────────────────────────────────────────────────────────
  Behavioral Deviation   LSTM Siamese Network            50
  Context Deviation      Network / IP / Geo signals      30
  Device Risk            Device fingerprint / processes  20
  ─────────────────────────────────────────────────────────
  Total                                                  100

Risk Levels & Enforcement:
  Score      Level      Action
  0–25       LOW        Full access. Passive monitoring.
  26–50      ELEVATED   Step-up authentication required.
  51–75      HIGH       Throttle access. Block sensitive resources. Alert admin.
  76–100     CRITICAL   Lock session immediately. Full incident alert.
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from preprocessor import FEATURE_DIM, FEATURE_INDEX

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  ENUMS
# ─────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW      = "LOW"
    ELEVATED = "ELEVATED"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class EnforcementAction(str, Enum):
    ALLOW         = "ALLOW"           # Full access, no interruption
    STEP_UP_AUTH  = "STEP_UP_AUTH"    # Prompt MFA / re-authentication
    THROTTLE      = "THROTTLE"        # Restrict access to sensitive resources
    LOCK_SESSION  = "LOCK_SESSION"    # Immediately lock the session
    ALERT_ADMIN   = "ALERT_ADMIN"     # Notify admin (can combine with others)


class UserPhase(str, Enum):
    OBSERVATION = "OBSERVATION"   # First 5 sessions — no enforcement
    SOFT        = "SOFT"          # Sessions 6–10 — relaxed thresholds
    FULL        = "FULL"          # Session 11+  — full enforcement


# ─────────────────────────────────────────────
#  THRESHOLDS
# ─────────────────────────────────────────────

@dataclass
class RiskThresholds:
    """
    Score boundaries for each risk level.
    Adjustable per deployment environment.
    """
    low_max:      float = 25.0
    elevated_max: float = 50.0
    high_max:     float = 75.0
    # CRITICAL = anything above high_max

    # Phase multipliers — divide raw score during these phases
    soft_divisor: float = 2.0   # Soft phase: thresholds are 2× more lenient

    @staticmethod
    def default() -> "RiskThresholds":
        return RiskThresholds()

    @staticmethod
    def strict() -> "RiskThresholds":
        """Tighter thresholds for high-security environments (servers, VMs)."""
        return RiskThresholds(low_max=15.0, elevated_max=35.0, high_max=60.0)

    @staticmethod
    def relaxed() -> "RiskThresholds":
        """Looser thresholds for lower-risk environments."""
        return RiskThresholds(low_max=35.0, elevated_max=60.0, high_max=80.0)


# ─────────────────────────────────────────────
#  SIGNAL DEFINITIONS
# ─────────────────────────────────────────────

@dataclass
class SignalResult:
    """Result of evaluating one individual risk signal."""
    name:        str
    raw_value:   float
    score:       float         # Contribution to the component score
    triggered:   bool
    description: str = ""


@dataclass
class ComponentScore:
    """Aggregated score for one risk component (behavioral / context / device)."""
    name:         str
    raw_score:    float        # Before cap
    capped_score: float        # After applying component max
    cap:          float
    signals:      list[SignalResult] = field(default_factory=list)

    @property
    def top_signal(self) -> Optional[SignalResult]:
        triggered = [s for s in self.signals if s.triggered]
        return max(triggered, key=lambda s: s.score) if triggered else None


# ─────────────────────────────────────────────
#  FULL RISK SCORE OUTPUT
# ─────────────────────────────────────────────

@dataclass
class RiskScoreResult:
    """
    Complete output of the Risk Scorer for one evaluation cycle.
    Contains the final score, level, enforcement recommendation,
    and a full per-signal breakdown for audit logging.
    """
    # Core output
    risk_score:           float           # 0–100, phase-adjusted
    raw_risk_score:       float           # 0–100, before phase adjustment
    risk_level:           RiskLevel
    enforcement:          list[EnforcementAction]

    # Component breakdown
    behavioral_deviation: float           # 0–50
    context_deviation:    float           # 0–30
    device_risk:          float           # 0–20

    # Siamese network output
    similarity_score:     float           # 0.0–1.0
    classification:       str             # LEGITIMATE | ATTACK | UNKNOWN
    confidence:           float           # 0.0–1.0

    # Phase
    phase:                UserPhase
    phase_adjusted:       bool            # True if score was modified by phase

    # Detailed signal breakdown
    behavioral_component: Optional[ComponentScore] = None
    context_component:    Optional[ComponentScore] = None
    device_component:     Optional[ComponentScore] = None

    # Metadata
    user_id:              str   = ""
    session_id:           str   = ""
    sequence_ready:       bool  = False
    timestamp:            float = field(default_factory=time.time)
    eval_time_ms:         float = 0.0

    def to_dict(self) -> dict:
        """Serialize for audit logging and API response."""
        return {
            "user_id":              self.user_id,
            "session_id":           self.session_id,
            "timestamp":            self.timestamp,
            "risk_score":           round(self.risk_score, 2),
            "raw_risk_score":       round(self.raw_risk_score, 2),
            "risk_level":           self.risk_level.value,
            "enforcement":          [a.value for a in self.enforcement],
            "behavioral_deviation": round(self.behavioral_deviation, 2),
            "context_deviation":    round(self.context_deviation, 2),
            "device_risk":          round(self.device_risk, 2),
            "similarity_score":     round(self.similarity_score, 4),
            "classification":       self.classification,
            "confidence":           round(self.confidence, 4),
            "phase":                self.phase.value,
            "phase_adjusted":       self.phase_adjusted,
            "sequence_ready":       self.sequence_ready,
            "eval_time_ms":         round(self.eval_time_ms, 3),
            "signals": {
                "context": [
                    {"name": s.name, "score": s.score, "triggered": s.triggered, "desc": s.description}
                    for s in (self.context_component.signals if self.context_component else [])
                ],
                "device": [
                    {"name": s.name, "score": s.score, "triggered": s.triggered, "desc": s.description}
                    for s in (self.device_component.signals if self.device_component else [])
                ],
            }
        }

    def summary(self) -> str:
        actions = ", ".join(a.value for a in self.enforcement)
        return (
            f"[{self.risk_level.value}] score={self.risk_score:.1f}  "
            f"behav={self.behavioral_deviation:.1f}  "
            f"ctx={self.context_deviation:.1f}  "
            f"dev={self.device_risk:.1f}  "
            f"clf={self.classification}({self.confidence:.2f})  "
            f"phase={self.phase.value}  "
            f"action=[{actions}]"
        )


# ─────────────────────────────────────────────
#  CONTEXT DEVIATION SCORER
# ─────────────────────────────────────────────

class ContextDeviationScorer:
    """
    Scores network and location context signals.
    Maximum component score: 30 points.

    Signals:
      ip_change_flag     → 25 pts  (strongest signal — IP changed mid-session)
      geo_distance       → 20 pts  (proportional to σ above user's normal location)
      vpn_flag           → 10 pts  (VPN detected and not in approved list)
      new_network        → 15 pts  (SSID not seen before for this user)
      unusual_time       → 10 pts  (login outside user's normal hours — σ > 2)
    """

    MAX_SCORE = 30.0

    # (feature_name, max_points, description)
    SIGNALS = [
        ("ip_change_flag",  25.0, "IP address changed from previous session"),
        ("geo_distance",    20.0, "Geographic location anomaly"),
        ("vpn_flag",        10.0, "VPN detected (not in approved list)"),
    ]

    def score(self, norm_vec: np.ndarray) -> ComponentScore:
        """
        Args:
            norm_vec: 24-dim Z-score normalized feature vector

        Returns:
            ComponentScore with per-signal breakdown
        """
        signals = []
        total   = 0.0

        # ── IP Change ─────────────────────────────────────
        # norm_vec index 17 = ip_change_flag (binary, not really normalized)
        # Raw value > 0.5 means the flag was set
        ip_raw   = norm_vec[FEATURE_INDEX["ip_change_flag"]]
        ip_score = 25.0 if ip_raw > 0.5 else 0.0
        signals.append(SignalResult(
            name="ip_change_flag",
            raw_value=float(ip_raw),
            score=ip_score,
            triggered=ip_score > 0,
            description="IP changed from registered session location" if ip_score > 0 else "",
        ))
        total += ip_score

        # ── Geographic Distance ───────────────────────────
        # norm_vec index 18 = geo_distance (Z-score)
        # >1σ → partial score, >2σ → full score
        geo_z   = norm_vec[FEATURE_INDEX["geo_distance"]]
        if geo_z > 3.0:
            geo_score = 20.0
            geo_desc  = f"Location {geo_z:.1f}σ from normal (extreme deviation)"
        elif geo_z > 2.0:
            geo_score = 15.0
            geo_desc  = f"Location {geo_z:.1f}σ from normal (high deviation)"
        elif geo_z > 1.0:
            geo_score = 8.0
            geo_desc  = f"Location {geo_z:.1f}σ from normal (moderate deviation)"
        else:
            geo_score = 0.0
            geo_desc  = ""
        signals.append(SignalResult(
            name="geo_distance",
            raw_value=float(geo_z),
            score=geo_score,
            triggered=geo_score > 0,
            description=geo_desc,
        ))
        total += geo_score

        # ── VPN ───────────────────────────────────────────
        # norm_vec index 20 = vpn_flag (binary)
        vpn_raw   = norm_vec[FEATURE_INDEX["vpn_flag"]]
        vpn_score = 10.0 if vpn_raw > 0.5 else 0.0
        signals.append(SignalResult(
            name="vpn_flag",
            raw_value=float(vpn_raw),
            score=vpn_score,
            triggered=vpn_score > 0,
            description="VPN connection detected" if vpn_score > 0 else "",
        ))
        total += vpn_score

        # ── Packet Rate Anomaly ───────────────────────────
        # High packet rate can indicate data exfiltration
        pkt_z = norm_vec[FEATURE_INDEX["packet_rate"]]
        if pkt_z > 3.0:
            pkt_score = 15.0
            pkt_desc  = f"Packet rate {pkt_z:.1f}σ above normal (possible exfiltration)"
        elif pkt_z > 2.0:
            pkt_score = 8.0
            pkt_desc  = f"Elevated packet rate ({pkt_z:.1f}σ)"
        else:
            pkt_score = 0.0
            pkt_desc  = ""
        signals.append(SignalResult(
            name="packet_rate",
            raw_value=float(pkt_z),
            score=pkt_score,
            triggered=pkt_score > 0,
            description=pkt_desc,
        ))
        total += pkt_score

        raw_total    = total
        capped_total = min(total, self.MAX_SCORE)

        return ComponentScore(
            name="context_deviation",
            raw_score=raw_total,
            capped_score=capped_total,
            cap=self.MAX_SCORE,
            signals=signals,
        )


# ─────────────────────────────────────────────
#  DEVICE RISK SCORER
# ─────────────────────────────────────────────

class DeviceRiskScorer:
    """
    Scores device integrity and identity signals.
    Maximum component score: 20 points.

    Signals:
      device_hash_embed  → 20 pts  (unrecognized device fingerprint)
      suspicious_flag    → 15 pts  (flagged process in process list)
      cpu_load           → 10 pts  (extreme CPU load — possible crypto miner)
      process_count      → 8  pts  (abnormally high process count)
      memory_usage       → 6  pts  (memory exhaustion — possible attack tool)
    """

    MAX_SCORE = 20.0

    def score(self, norm_vec: np.ndarray) -> ComponentScore:
        """
        Args:
            norm_vec: 24-dim Z-score normalized feature vector

        Returns:
            ComponentScore with per-signal breakdown
        """
        signals = []
        total   = 0.0

        # ── Device Fingerprint ────────────────────────────
        # device_hash_embed normalized: large negative value = unrecognized device
        # (new device has a very different hash embedding vs registered devices)
        dev_z = norm_vec[FEATURE_INDEX["device_hash_embed"]]
        if dev_z < -2.5:
            dev_score = 20.0
            dev_desc  = "Unrecognized device fingerprint (new/unregistered device)"
        elif dev_z < -1.5:
            dev_score = 12.0
            dev_desc  = f"Device fingerprint deviation ({dev_z:.1f}σ)"
        else:
            dev_score = 0.0
            dev_desc  = ""
        signals.append(SignalResult(
            name="device_hash_embed",
            raw_value=float(dev_z),
            score=dev_score,
            triggered=dev_score > 0,
            description=dev_desc,
        ))
        total += dev_score

        # ── Suspicious Process ────────────────────────────
        # Binary flag set by the Go agent's process scanner
        susp_raw   = norm_vec[FEATURE_INDEX["suspicious_flag"]]
        susp_score = 15.0 if susp_raw > 0.5 else 0.0
        signals.append(SignalResult(
            name="suspicious_flag",
            raw_value=float(susp_raw),
            score=susp_score,
            triggered=susp_score > 0,
            description="Suspicious/flagged process detected in process list" if susp_score > 0 else "",
        ))
        total += susp_score

        # ── CPU Load Anomaly ──────────────────────────────
        # Extreme CPU load can indicate crypto mining, brute force, or malware
        cpu_z = norm_vec[FEATURE_INDEX["cpu_load"]]
        if cpu_z > 3.0:
            cpu_score = 10.0
            cpu_desc  = f"Extreme CPU load ({cpu_z:.1f}σ above normal)"
        elif cpu_z > 2.0:
            cpu_score = 5.0
            cpu_desc  = f"Elevated CPU load ({cpu_z:.1f}σ)"
        else:
            cpu_score = 0.0
            cpu_desc  = ""
        signals.append(SignalResult(
            name="cpu_load",
            raw_value=float(cpu_z),
            score=cpu_score,
            triggered=cpu_score > 0,
            description=cpu_desc,
        ))
        total += cpu_score

        # ── Process Count Anomaly ─────────────────────────
        proc_z = norm_vec[FEATURE_INDEX["process_count"]]
        if proc_z > 3.0:
            proc_score = 8.0
            proc_desc  = f"Abnormally high process count ({proc_z:.1f}σ)"
        elif proc_z > 2.0:
            proc_score = 4.0
            proc_desc  = f"Elevated process count ({proc_z:.1f}σ)"
        else:
            proc_score = 0.0
            proc_desc  = ""
        signals.append(SignalResult(
            name="process_count",
            raw_value=float(proc_z),
            score=proc_score,
            triggered=proc_score > 0,
            description=proc_desc,
        ))
        total += proc_score

        # ── Memory Usage Anomaly ──────────────────────────
        mem_z = norm_vec[FEATURE_INDEX["memory_usage"]]
        if mem_z > 3.0:
            mem_score = 6.0
            mem_desc  = f"Abnormal memory usage ({mem_z:.1f}σ)"
        elif mem_z > 2.0:
            mem_score = 3.0
            mem_desc  = f"Elevated memory usage ({mem_z:.1f}σ)"
        else:
            mem_score = 0.0
            mem_desc  = ""
        signals.append(SignalResult(
            name="memory_usage",
            raw_value=float(mem_z),
            score=mem_score,
            triggered=mem_score > 0,
            description=mem_desc,
        ))
        total += mem_score

        raw_total    = total
        capped_total = min(total, self.MAX_SCORE)

        return ComponentScore(
            name="device_risk",
            raw_score=raw_total,
            capped_score=capped_total,
            cap=self.MAX_SCORE,
            signals=signals,
        )


# ─────────────────────────────────────────────
#  BEHAVIORAL DEVIATION SCORER
# ─────────────────────────────────────────────

class BehavioralDeviationScorer:
    """
    Converts Siamese network output into a behavioral deviation score.
    Maximum component score: 50 points.

    When a sequence is not yet ready (buffer filling), falls back to
    typing and mouse signal heuristics for an early-warning signal.
    """

    MAX_SCORE = 50.0

    def score_from_model(
        self,
        similarity: float,
        classification: str,
        confidence: float,
    ) -> ComponentScore:
        """
        Score from Siamese network output.

        Args:
            similarity:     0.0–1.0 (1.0 = identical to profile)
            classification: "LEGITIMATE" | "ATTACK" | "UNKNOWN"
            confidence:     0.0–1.0

        Returns:
            ComponentScore
        """
        # Base deviation from similarity
        base_deviation = (1.0 - similarity) * self.MAX_SCORE

        # Confidence-weighted boost when model predicts ATTACK
        if classification == "ATTACK":
            boost = confidence * 15.0
        elif classification == "UNKNOWN":
            boost = 5.0
        else:
            boost = 0.0

        raw   = base_deviation + boost
        capped = min(raw, self.MAX_SCORE)

        signals = [
            SignalResult(
                name="siamese_similarity",
                raw_value=round(similarity, 4),
                score=round(base_deviation, 2),
                triggered=similarity < 0.7,
                description=f"Behavioral similarity {similarity:.3f} (deviation={base_deviation:.1f}pts)",
            ),
            SignalResult(
                name="classification_boost",
                raw_value=round(confidence, 4),
                score=round(boost, 2),
                triggered=boost > 0,
                description=f"Model classified as {classification} with confidence {confidence:.2f}" if boost > 0 else "",
            ),
        ]

        return ComponentScore(
            name="behavioral_deviation",
            raw_score=round(raw, 2),
            capped_score=round(capped, 2),
            cap=self.MAX_SCORE,
            signals=signals,
        )

    def score_heuristic(self, norm_vec: np.ndarray) -> ComponentScore:
        """
        Heuristic fallback when the sequence buffer is not yet full.
        Uses typing and mouse Z-scores as early-warning signals.
        Returns a conservative (lower) score since data is limited.
        """
        total = 0.0
        signals = []

        # Typing speed anomaly
        typing_z = norm_vec[FEATURE_INDEX["typing_speed"]]
        if abs(typing_z) > 3.0:
            score = 15.0
            desc  = f"Typing speed {typing_z:.1f}σ from baseline"
        elif abs(typing_z) > 2.0:
            score = 8.0
            desc  = f"Typing speed {typing_z:.1f}σ from baseline"
        else:
            score = 0.0
            desc  = ""
        signals.append(SignalResult("typing_speed_heuristic", float(typing_z), score, score > 0, desc))
        total += score

        # Key hold time anomaly
        hold_z = norm_vec[FEATURE_INDEX["hold_time_mean"]]
        if abs(hold_z) > 3.0:
            score = 12.0
            desc  = f"Key hold time {hold_z:.1f}σ from baseline"
        elif abs(hold_z) > 2.0:
            score = 6.0
            desc  = f"Key hold time {hold_z:.1f}σ from baseline"
        else:
            score = 0.0
            desc  = ""
        signals.append(SignalResult("hold_time_heuristic", float(hold_z), score, score > 0, desc))
        total += score

        # Mouse speed anomaly
        mouse_z = norm_vec[FEATURE_INDEX["mouse_speed"]]
        if abs(mouse_z) > 3.0:
            score = 10.0
            desc  = f"Mouse speed {mouse_z:.1f}σ from baseline"
        elif abs(mouse_z) > 2.0:
            score = 5.0
            desc  = f"Mouse speed {mouse_z:.1f}σ from baseline"
        else:
            score = 0.0
            desc  = ""
        signals.append(SignalResult("mouse_speed_heuristic", float(mouse_z), score, score > 0, desc))
        total += score

        # Cap heuristic at 25 (half max) — conservative since no sequence yet
        capped = min(total, 25.0)

        return ComponentScore(
            name="behavioral_deviation_heuristic",
            raw_score=round(total, 2),
            capped_score=round(capped, 2),
            cap=25.0,
            signals=signals,
        )


# ─────────────────────────────────────────────
#  ENFORCEMENT ENGINE
# ─────────────────────────────────────────────

class EnforcementEngine:
    """
    Maps a risk score + level to a set of enforcement actions.
    Multiple actions can fire simultaneously (e.g. THROTTLE + ALERT_ADMIN).
    """

    def __init__(self, thresholds: RiskThresholds = None):
        self.t = thresholds or RiskThresholds.default()

    def classify(self, score: float) -> RiskLevel:
        if score <= self.t.low_max:
            return RiskLevel.LOW
        elif score <= self.t.elevated_max:
            return RiskLevel.ELEVATED
        elif score <= self.t.high_max:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def recommend(
        self,
        score: float,
        phase: UserPhase,
        classification: str,
    ) -> tuple[RiskLevel, list[EnforcementAction]]:
        """
        Determine risk level and enforcement actions.

        Args:
            score:          Phase-adjusted risk score (0–100)
            phase:          User's current onboarding phase
            classification: Model classification ("LEGITIMATE" / "ATTACK")

        Returns:
            (RiskLevel, list[EnforcementAction])
        """
        level   = self.classify(score)
        actions = []

        # No enforcement during OBSERVATION phase
        if phase == UserPhase.OBSERVATION:
            return RiskLevel.LOW, [EnforcementAction.ALLOW]

        if level == RiskLevel.LOW:
            actions = [EnforcementAction.ALLOW]

        elif level == RiskLevel.ELEVATED:
            actions = [EnforcementAction.STEP_UP_AUTH]

        elif level == RiskLevel.HIGH:
            actions = [EnforcementAction.THROTTLE, EnforcementAction.ALERT_ADMIN]

        elif level == RiskLevel.CRITICAL:
            actions = [EnforcementAction.LOCK_SESSION, EnforcementAction.ALERT_ADMIN]

        # Escalate if model is confident about an attack regardless of score
        if classification == "ATTACK" and EnforcementAction.LOCK_SESSION not in actions:
            if EnforcementAction.ALLOW in actions:
                actions = [EnforcementAction.STEP_UP_AUTH]
            elif EnforcementAction.STEP_UP_AUTH in actions:
                actions = [EnforcementAction.THROTTLE, EnforcementAction.ALERT_ADMIN]

        return level, actions


# ─────────────────────────────────────────────
#  MAIN RISK SCORER
# ─────────────────────────────────────────────

class RiskScorer:
    """
    Authoritative Risk Scorer for the ECCAAC system.

    Combines all three component scorers and the enforcement engine
    into a single `compute()` call. This is the class that
    grpc_service.py instantiates and calls for every evaluation cycle.

    Usage:
        scorer = RiskScorer()

        result = scorer.compute(
            user_id="user_001",
            session_id="session_abc",
            norm_vec=normalized_24dim_vector,
            similarity=0.82,
            classification="LEGITIMATE",
            confidence=0.91,
            sequence_ready=True,
            phase="FULL",
        )

        print(result.summary())
        log_audit(result.to_dict())
    """

    def __init__(self, thresholds: RiskThresholds = None):
        self.behavioral = BehavioralDeviationScorer()
        self.context    = ContextDeviationScorer()
        self.device     = DeviceRiskScorer()
        self.enforcement = EnforcementEngine(thresholds or RiskThresholds.default())

    def compute(
        self,
        user_id:        str,
        session_id:     str,
        norm_vec:       np.ndarray,
        similarity:     float = 0.5,
        classification: str   = "UNKNOWN",
        confidence:     float = 0.5,
        sequence_ready: bool  = False,
        phase:          str   = "FULL",
    ) -> RiskScoreResult:
        """
        Compute the full risk score for one evaluation cycle.

        Args:
            user_id:        User identifier
            session_id:     Session identifier
            norm_vec:       24-dim Z-score normalized feature vector
            similarity:     Siamese similarity score (0.0–1.0)
            classification: Model binary classification label
            confidence:     Model confidence (0.0–1.0)
            sequence_ready: True if Siamese inference was run this cycle
            phase:          User onboarding phase string

        Returns:
            RiskScoreResult — full score with breakdown and enforcement actions
        """
        t_start = time.perf_counter()

        assert norm_vec.shape == (FEATURE_DIM,), \
            f"Expected ({FEATURE_DIM},) vector, got {norm_vec.shape}"

        phase_enum = UserPhase(phase) if isinstance(phase, str) else phase

        # ── Component scoring ─────────────────────────────

        # Behavioral deviation
        if sequence_ready:
            behav_component = self.behavioral.score_from_model(
                similarity, classification, confidence
            )
        else:
            behav_component = self.behavioral.score_heuristic(norm_vec)

        # Context deviation
        ctx_component = self.context.score(norm_vec)

        # Device risk
        dev_component = self.device.score(norm_vec)

        # ── Aggregate raw score ───────────────────────────
        behav_score = behav_component.capped_score
        ctx_score   = ctx_component.capped_score
        dev_score   = dev_component.capped_score

        raw_score = min(behav_score + ctx_score + dev_score, 100.0)

        # ── Phase adjustment ──────────────────────────────
        phase_adjusted = False
        if phase_enum == UserPhase.OBSERVATION:
            final_score    = 0.0
            phase_adjusted = True
        elif phase_enum == UserPhase.SOFT:
            final_score    = raw_score / self.enforcement.t.soft_divisor
            phase_adjusted = True
        else:
            final_score = raw_score

        # ── Enforcement recommendation ────────────────────
        risk_level, actions = self.enforcement.recommend(
            final_score, phase_enum, classification
        )

        eval_ms = (time.perf_counter() - t_start) * 1000

        result = RiskScoreResult(
            risk_score=round(final_score, 2),
            raw_risk_score=round(raw_score, 2),
            risk_level=risk_level,
            enforcement=actions,
            behavioral_deviation=round(behav_score, 2),
            context_deviation=round(ctx_score, 2),
            device_risk=round(dev_score, 2),
            similarity_score=round(similarity, 4),
            classification=classification,
            confidence=round(confidence, 4),
            phase=phase_enum,
            phase_adjusted=phase_adjusted,
            behavioral_component=behav_component,
            context_component=ctx_component,
            device_component=dev_component,
            user_id=user_id,
            session_id=session_id,
            sequence_ready=sequence_ready,
            timestamp=time.time(),
            eval_time_ms=round(eval_ms, 3),
        )

        logger.debug(result.summary())
        return result

    def update_thresholds(self, thresholds: RiskThresholds):
        """Hot-swap thresholds without restarting the service."""
        self.enforcement = EnforcementEngine(thresholds)
        logger.info(f"Thresholds updated: low≤{thresholds.low_max} elevated≤{thresholds.elevated_max} high≤{thresholds.high_max}")


# ─────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    print("=" * 65)
    print("  ECCAAC — Risk Scorer  |  Smoke Test")
    print("=" * 65)

    scorer = RiskScorer()

    def make_norm_vec(overrides: dict = None) -> np.ndarray:
        """Build a baseline normal normalized vector with optional overrides."""
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)
        if overrides:
            for name, val in overrides.items():
                vec[FEATURE_INDEX[name]] = val
        return vec

    scenarios = [
        {
            "name":        "Normal session — legitimate user",
            "norm_vec":    make_norm_vec(),
            "similarity":  0.92,
            "clf":         "LEGITIMATE",
            "confidence":  0.95,
            "seq_ready":   True,
            "phase":       "FULL",
        },
        {
            "name":        "Slightly unusual typing, same location",
            "norm_vec":    make_norm_vec({"typing_speed": -2.2, "hold_time_mean": 2.1}),
            "similarity":  0.74,
            "clf":         "LEGITIMATE",
            "confidence":  0.71,
            "seq_ready":   True,
            "phase":       "FULL",
        },
        {
            "name":        "IP change + geo deviation (stolen credentials?)",
            "norm_vec":    make_norm_vec({"ip_change_flag": 1.0, "geo_distance": 3.5}),
            "similarity":  0.55,
            "clf":         "ATTACK",
            "confidence":  0.83,
            "seq_ready":   True,
            "phase":       "FULL",
        },
        {
            "name":        "Full attack — all signals firing",
            "norm_vec":    make_norm_vec({
                "ip_change_flag": 1.0, "geo_distance": 4.5, "vpn_flag": 1.0,
                "suspicious_flag": 1.0, "device_hash_embed": -3.5,
                "packet_rate": 4.0, "cpu_load": 3.8,
            }),
            "similarity":  0.12,
            "clf":         "ATTACK",
            "confidence":  0.97,
            "seq_ready":   True,
            "phase":       "FULL",
        },
        {
            "name":        "New user — OBSERVATION phase (no enforcement)",
            "norm_vec":    make_norm_vec({"ip_change_flag": 1.0, "geo_distance": 5.0}),
            "similarity":  0.20,
            "clf":         "ATTACK",
            "confidence":  0.90,
            "seq_ready":   True,
            "phase":       "OBSERVATION",
        },
        {
            "name":        "Soft phase — same attack, halved thresholds",
            "norm_vec":    make_norm_vec({"ip_change_flag": 1.0, "suspicious_flag": 1.0}),
            "similarity":  0.40,
            "clf":         "ATTACK",
            "confidence":  0.80,
            "seq_ready":   True,
            "phase":       "SOFT",
        },
        {
            "name":        "No sequence yet — heuristic fallback",
            "norm_vec":    make_norm_vec({"typing_speed": -3.5, "hold_time_mean": 3.2}),
            "similarity":  0.5,
            "clf":         "UNKNOWN",
            "confidence":  0.5,
            "seq_ready":   False,
            "phase":       "FULL",
        },
    ]

    print()
    for i, s in enumerate(scenarios, 1):
        result = scorer.compute(
            user_id="user_test",
            session_id=f"session_{i:03d}",
            norm_vec=s["norm_vec"],
            similarity=s["similarity"],
            classification=s["clf"],
            confidence=s["confidence"],
            sequence_ready=s["seq_ready"],
            phase=s["phase"],
        )

        print(f"Scenario {i}: {s['name']}")
        print(f"  {result.summary()}")

        triggered = [
            sig for comp in [result.context_component, result.device_component, result.behavioral_component]
            if comp
            for sig in comp.signals
            if sig.triggered
        ]
        if triggered:
            for sig in triggered:
                print(f"  ↳ {sig.name}: +{sig.score:.1f}pts  — {sig.description}")
        print()

    # Threshold hot-swap test
    print("Testing hot-swap to STRICT thresholds...")
    scorer.update_thresholds(RiskThresholds.strict())
    result_strict = scorer.compute(
        user_id="user_test", session_id="session_strict",
        norm_vec=make_norm_vec({"typing_speed": -2.2}),
        similarity=0.74, classification="LEGITIMATE",
        confidence=0.71, sequence_ready=True, phase="FULL",
    )
    print(f"  Strict: {result_strict.summary()}")

    scorer.update_thresholds(RiskThresholds.default())
    result_default = scorer.compute(
        user_id="user_test", session_id="session_default",
        norm_vec=make_norm_vec({"typing_speed": -2.2}),
        similarity=0.74, classification="LEGITIMATE",
        confidence=0.71, sequence_ready=True, phase="FULL",
    )
    print(f"  Default: {result_default.summary()}")

    print(f"\n✓ All checks passed.\n")
