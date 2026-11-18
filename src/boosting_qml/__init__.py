"""Core package for boosting-vs-PEC experiments."""

from .config import (
    BoostingStopTargetConfig,
    EMRunArtifactConfig,
    EMTargetAggregationConfig,
    Phase0ExperimentConfig,
    RoundBudgetBoundsConfig,
)
from .types import (
    BinaryDataset,
    BinaryDatasetSplit,
    BoostingRoundResult,
    BoostingStopSummary,
    EMRunSummary,
    EMTargetAggregationSummary,
    RoundBudgetBoundFamily,
)

__all__ = [
    'BinaryDataset',
    'BinaryDatasetSplit',
    'BoostingRoundResult',
    'BoostingStopSummary',
    'BoostingStopTargetConfig',
    'EMRunArtifactConfig',
    'EMRunSummary',
    'EMTargetAggregationConfig',
    'EMTargetAggregationSummary',
    'Phase0ExperimentConfig',
    'RoundBudgetBoundFamily',
    'RoundBudgetBoundsConfig',
]
