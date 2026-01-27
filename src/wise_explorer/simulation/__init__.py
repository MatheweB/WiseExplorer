"""
Simulation module - parallel game execution and training.

Provides the infrastructure for running many game simulations in parallel
and collecting results for training.
"""

from wise_explorer.simulation.jobs import GameJob, JobResult, MoveRecord
from wise_explorer.simulation.runner import SimulationRunner, DEFAULT_WORKER_COUNT
from wise_explorer.simulation.training import run_training, run_training_interleaved

__all__ = [
    "GameJob",
    "JobResult",
    "MoveRecord",
    "SimulationRunner",
    "DEFAULT_WORKER_COUNT",
    "run_training",
    "run_training_interleaved",
]