from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from omegaconf import MISSING, DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException


@dataclass(slots=True)
class EnvConfig:
    id: str = MISSING
    render_mode: str | None = None


@dataclass(slots=True)
class RandomAgentConfig:
    pass


@dataclass(slots=True)
class ReinforceAgentConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    hidden_dim: int = 128
    batch_episodes: int = 4


@dataclass(slots=True)
class DQNAgentConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    hidden_dim: int = 128
    buffer_capacity: int = 50_000
    batch_size: int = 64
    target_update_frequency: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    min_buffer_size: int = 1_000


@dataclass(slots=True)
class AgentConfig:
    name: str = MISSING
    seed: int | None = None
    random: RandomAgentConfig = field(default_factory=RandomAgentConfig)
    reinforce: ReinforceAgentConfig = field(default_factory=ReinforceAgentConfig)
    dqn: DQNAgentConfig = field(default_factory=DQNAgentConfig)


@dataclass(slots=True)
class TrainingConfig:
    episodes: int = MISSING
    max_steps_per_episode: int = MISSING
    seed: int = MISSING
    log_every: int = 10
    update_every: int = 1


@dataclass(slots=True)
class EvaluationConfig:
    enabled: bool = True
    episodes: int = 3
    frequency: int = 5
    render_mode: str | None = None


@dataclass(slots=True)
class CheckpointConfig:
    enabled: bool = True
    directory: str = "checkpoints"
    frequency: int = 5
    save_best: bool = True


@dataclass(slots=True)
class WandbConfig:
    enabled: bool = False
    project: str = "rl-lunar-lander"
    entity: str | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OptimizeConfig:
    enabled: bool = False
    project: str = "rl-lunar-lander"
    n_trials: int = 20
    direction: str = "maximize"
    study_name: str | None = None


@dataclass(slots=True)
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    agent_config: str | None = None


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)

    try:
        schema = OmegaConf.structured(ExperimentConfig)
        root_config = _load_config(config_path)
        agent_config = _load_agent_config(root_config, config_path)
        merged_config = OmegaConf.merge(schema, root_config, agent_config)
        return cast(ExperimentConfig, OmegaConf.to_object(merged_config))
    except OmegaConfBaseException as exc:
        msg = f"Invalid config at {config_path}: {exc}"
        raise ValueError(msg) from exc


def _load_config(path: Path) -> DictConfig:
    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        msg = f"Config file must contain a mapping: {path}"
        raise ValueError(msg)
    return loaded


def _load_agent_config(root_config: DictConfig, config_path: Path) -> DictConfig:
    agent_config_path = OmegaConf.select(root_config, "agent_config")
    if agent_config_path is None:
        return OmegaConf.create({})

    resolved_path = (config_path.parent / str(agent_config_path)).resolve()
    return _load_config(resolved_path)
