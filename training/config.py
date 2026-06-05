from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from omegaconf import MISSING, OmegaConf
from omegaconf.errors import OmegaConfBaseException


@dataclass(slots=True)
class EnvConfig:
    id: str = MISSING
    render_mode: str | None = None


@dataclass(slots=True)
class AgentConfig:
    name: str = MISSING
    seed: int | None = None


@dataclass(slots=True)
class TrainingConfig:
    episodes: int = MISSING
    max_steps_per_episode: int = MISSING
    seed: int = MISSING
    log_every: int = 10


@dataclass(slots=True)
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)

    try:
        schema = OmegaConf.structured(ExperimentConfig)
        file_config = OmegaConf.load(config_path)
        merged_config = OmegaConf.merge(schema, file_config)
        return cast(ExperimentConfig, OmegaConf.to_object(merged_config))
    except OmegaConfBaseException as exc:
        msg = f"Invalid config at {config_path}: {exc}"
        raise ValueError(msg) from exc
