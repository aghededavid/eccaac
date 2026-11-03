"""
ECCAAC — Risk Scoring Engine
Python Service Package

Exports the core public API for the risk engine.
"""

from risk_engine.core.siamese_network import (
    SiameseNetwork,
    SiameseConfig,
    build_model,
    save_model,
    load_model,
)
from risk_engine.core.preprocessor import (
    FeaturePreprocessor,
    RawFeatures,
    UserStats,
    FEATURE_DIM,
    SEQUENCE_LEN,
    FEATURE_INDEX,
)
from risk_engine.core.risk_scorer import (
    RiskScorer,
    RiskScoreResult,
    RiskLevel,
    RiskThresholds,
    EnforcementAction,
    UserPhase,
)
from risk_engine.core.trainer import (
    SiameseTrainer,
    TrainingConfig,
    BehavioralPairDataset,
)

__version__ = "1.0.0"
__all__ = [
    "SiameseNetwork", "SiameseConfig", "build_model", "save_model", "load_model",
    "FeaturePreprocessor", "RawFeatures", "UserStats",
    "FEATURE_DIM", "SEQUENCE_LEN", "FEATURE_INDEX",
    "RiskScorer", "RiskScoreResult", "RiskLevel", "RiskThresholds",
    "EnforcementAction", "UserPhase",
    "SiameseTrainer", "TrainingConfig", "BehavioralPairDataset",
]
