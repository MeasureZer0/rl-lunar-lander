from __future__ import annotations

import copy
import dataclasses
import json
import shutil
from pathlib import Path
from statistics import mean
from typing import Protocol, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.importance
import optuna.visualization
import plotly.graph_objects
import wandb

from training.config import DoubleDQNAgentConfig, DQNAgentConfig, ExperimentConfig
from training.evaluate import evaluate_agent
from training.trainer import Trainer


class ArchitectureDebugAgent(Protocol):
    def parameter_count(self) -> int: ...

    def hidden_activation_snapshot(self, observations: np.ndarray) -> np.ndarray: ...


class NetworkArchitectureConfig(Protocol):
    hidden_layers: list[int]
    activation: str
    weight_init: str
    normalization: str | None
    normalization_position: str
    dropout: float


def run_trial(trial: optuna.Trial, base_config: ExperimentConfig) -> float:
    config = copy.deepcopy(base_config)
    mode = config.optimize.mode

    if mode not in {"hyperparameters", "architecture", "reward_shaping"}:
        msg = (
            f"Unsupported optimize.mode '{mode}'. Expected one of "
            "'hyperparameters', 'architecture', 'reward_shaping'."
        )
        raise ValueError(msg)

    if mode != "reward_shaping":
        if config.agent.name == "reinforce":
            if mode == "architecture":
                _sample_reinforce_architecture_config(trial, config)
            else:
                _sample_reinforce_hyperparameter_config(trial, config)
        elif config.agent.name == "dqn":
            if mode == "architecture":
                _sample_dqn_architecture_config(trial, config)
            else:
                _sample_dqn_hyperparameter_config(trial, config)
        elif config.agent.name == "double_dqn":
            if mode == "architecture":
                _sample_double_dqn_architecture_config(trial, config)
            else:
                _sample_double_dqn_hyperparameter_config(trial, config)

    if mode == "reward_shaping":
        if not config.reward_shaping.enabled:
            msg = (
                "optimize.mode is 'reward_shaping' but reward_shaping.enabled "
                "is false. Set reward_shaping.enabled: true so the sampled "
                "weights actually take effect during training and "
                "evaluation."
            )
            raise ValueError(msg)
        config.reward_shaping.w_align = trial.suggest_float("w_align", 0.0, 1.0)
        config.reward_shaping.w_tilt = trial.suggest_float("w_tilt", 0.0, 0.5)
        config.reward_shaping.w_soft = trial.suggest_float("w_soft", 0.0, 1.0)
        config.reward_shaping.w_hover = trial.suggest_float("w_hover", 0.0, 0.2)
        config.reward_shaping.w_leg_sym = trial.suggest_float("w_leg_sym", 0.0, 0.3)

    config.env.render_mode = None
    config.evaluation.render_mode = None
    config.wandb.name = f"{mode}-trial-{trial.number}"
    config.wandb.tags = [*config.wandb.tags, "optuna", mode]
    config.checkpoint.enabled = False

    trainer = Trainer(config)
    metrics = trainer.run()

    if not metrics or trainer.agent is None:
        return float("-inf")

    final_eval = evaluate_agent(
        agent=trainer.agent,
        env_config=config.env,
        training_config=config.training,
        evaluation_config=config.evaluation,
        seed_offset=20_000,
    )
    avg_reward = final_eval.avg_reward
    trial.set_user_attr("final_eval_avg_reward", final_eval.avg_reward)
    trial.set_user_attr("final_eval_avg_steps", final_eval.avg_steps)

    rewards = [item.total_reward for item in metrics[-10:]]
    raw_rewards = [item.raw_total_reward for item in metrics[-10:]]
    trial.set_user_attr("train_last10_avg_reward", mean(rewards))
    trial.set_user_attr("train_last10_avg_raw_reward", mean(raw_rewards))

    agent = trainer.agent
    if agent is not None and hasattr(agent, "parameter_count"):
        debug_agent = cast(ArchitectureDebugAgent, agent)
        parameter_count = debug_agent.parameter_count()
        trial.set_user_attr("parameter_count", parameter_count)
        if wandb.run is not None:
            wandb.log({"model/parameter_count": parameter_count})

    if (
        config.optimize.mode == "architecture"
        and agent is not None
        and hasattr(agent, "hidden_activation_snapshot")
    ):
        _log_hidden_activation_heatmap(cast(ArchitectureDebugAgent, agent), trial)

    all_rewards = [item.total_reward for item in metrics]
    all_raw_rewards = [item.raw_total_reward for item in metrics]
    _log_reward_distribution(all_rewards, trial)
    _log_smoothed_learning_curve(all_rewards, trial)
    _log_architecture_params_text(config, trial)
    _save_trial_artifact(trial, config, trainer)

    if (
        config.agent.name in {"dqn", "double_dqn"}
        and config.optimize.mode == "hyperparameters"
    ):
        _log_epsilon_decay_params(config, trial)

    if wandb.run is not None:
        wandb.log(
            {
                "optuna/objective_eval_avg_reward": final_eval.avg_reward,
                "optuna/final_eval_avg_steps": final_eval.avg_steps,
                "optuna/train_last10_avg_reward": mean(rewards),
                "optuna/train_last10_avg_raw_reward": mean(raw_rewards),
                "optuna/train_overall_avg_raw_reward": mean(all_raw_rewards),
            }
        )

    return avg_reward


def save_best_checkpoint(
    study: optuna.Study,
    base_config: ExperimentConfig,
    *,
    filename: str,
) -> Path:
    checkpoint_path = Path(base_config.optimize.checkpoint_directory) / filename
    trial_checkpoint = study.best_trial.user_attrs.get("checkpoint_path")
    if not isinstance(trial_checkpoint, str):
        msg = "Best trial did not record a checkpoint path."
        raise RuntimeError(msg)

    trial_checkpoint_path = Path(trial_checkpoint)
    if not trial_checkpoint_path.is_file():
        msg = f"Best trial checkpoint is missing: {trial_checkpoint_path}"
        raise RuntimeError(msg)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trial_checkpoint_path, checkpoint_path)

    trial_metadata = study.best_trial.user_attrs.get("metadata_path")
    if isinstance(trial_metadata, str):
        metadata_path = Path(trial_metadata)
        if metadata_path.is_file():
            shutil.copy2(
                metadata_path,
                checkpoint_path.with_suffix(".json"),
            )
    return checkpoint_path


def _save_trial_artifact(
    trial: optuna.Trial,
    config: ExperimentConfig,
    trainer: Trainer,
) -> None:
    if trainer.agent is None:
        return

    base_dir = (
        Path(config.optimize.checkpoint_directory)
        / "optuna_trials"
        / (config.optimize.study_name or f"optuna-study-{config.agent.name}")
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = base_dir / f"trial_{trial.number:04d}.pt"
    metadata_path = base_dir / f"trial_{trial.number:04d}.json"
    trainer.agent.save(checkpoint_path)
    trial.set_user_attr("checkpoint_path", str(checkpoint_path))
    trial.set_user_attr("metadata_path", str(metadata_path))
    metadata = {
        "trial_number": trial.number,
        "params": trial.params,
        "user_attrs": dict(trial.user_attrs),
        "config": dataclasses.asdict(config),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def log_architecture_plots(study: optuna.Study) -> None:
    if wandb.run is None:
        return

    xs: list[int] = []
    ys: list[float] = []
    for trial in study.trials:
        if trial.value is None:
            continue
        parameter_count = trial.user_attrs.get("parameter_count")
        if parameter_count is None:
            continue
        xs.append(int(parameter_count))
        ys.append(float(trial.value))

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(xs, ys)
    ax.set_xlabel("parameter_count")
    ax.set_ylabel("avg_reward")
    ax.set_title("Architecture size vs reward")
    ax.grid(alpha=0.25)
    wandb.log({"architecture/params_vs_avg_reward": wandb.Image(fig)})
    plt.close(fig)

    _log_trial_ranking_table(study)

    _log_reward_vs_trial_number(study)


def _sample_reinforce_hyperparameter_config(
    trial: optuna.Trial,
    config: ExperimentConfig,
) -> None:
    reinforce = config.agent.reinforce
    reinforce.learning_rate = trial.suggest_categorical(
        "learning_rate",
        [1e-4, 5e-4, 1e-3],
    )
    reinforce.gamma = trial.suggest_categorical("gamma", [0.95, 0.99, 0.999])
    # Widened from [1, 2, 4, 8]. REINFORCE's gradient variance only shrinks
    # as 1/sqrt(N) in the number of episodes per batch, so 1->4 episodes
    # barely halves the noise. The old upper bound of 8 also meant Optuna
    # could never discover whether an even larger batch helps -- it would
    # just always pick 8, the ceiling, with no signal about whether 16/32
    # would be better still.
    reinforce.batch_episodes = trial.suggest_categorical(
        "batch_episodes",
        [4, 8, 16, 32, 64],
    )
    reinforce.gradient_clip_max_norm = cast(
        float | None,
        trial.suggest_categorical(
            "gradient_clip_max_norm",
            [None, 1.0, 5.0, 10.0],
        ),
    )


def _sample_dqn_hyperparameter_config(
    trial: optuna.Trial,
    config: ExperimentConfig,
) -> None:
    dqn = config.agent.dqn
    dqn.learning_rate = trial.suggest_categorical(
        "learning_rate",
        [1e-4, 5e-4, 1e-3],
    )
    dqn.gamma = trial.suggest_categorical("gamma", [0.95, 0.99, 0.999])
    dqn.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    dqn.buffer_capacity = trial.suggest_categorical(
        "buffer_capacity",
        [5_000, 10_000, 50_000, 100_000],
    )
    dqn.min_buffer_size = min(dqn.min_buffer_size, dqn.buffer_capacity)
    dqn.min_buffer_size = max(dqn.min_buffer_size, dqn.batch_size)
    _sample_epsilon_schedule(trial, dqn)
    _sample_target_update(trial, dqn)
    _sample_lr_scheduler(trial, dqn)
    dqn.gradient_clip_max_norm = cast(
        float | None,
        trial.suggest_categorical(
            "gradient_clip_max_norm",
            [None, 1.0, 5.0, 10.0],
        ),
    )


def _sample_double_dqn_hyperparameter_config(
    trial: optuna.Trial,
    config: ExperimentConfig,
) -> None:
    double_dqn = config.agent.double_dqn
    double_dqn.learning_rate = trial.suggest_categorical(
        "learning_rate",
        [1e-4, 5e-4, 1e-3],
    )
    double_dqn.gamma = trial.suggest_categorical("gamma", [0.95, 0.99, 0.999])
    double_dqn.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    double_dqn.buffer_capacity = trial.suggest_categorical(
        "buffer_capacity",
        [5_000, 10_000, 50_000, 100_000],
    )
    double_dqn.min_buffer_size = min(
        double_dqn.min_buffer_size, double_dqn.buffer_capacity
    )
    double_dqn.min_buffer_size = max(double_dqn.min_buffer_size, double_dqn.batch_size)
    _sample_epsilon_schedule(trial, double_dqn)
    _sample_target_update(trial, double_dqn)
    _sample_lr_scheduler(trial, double_dqn)
    double_dqn.gradient_clip_max_norm = cast(
        float | None,
        trial.suggest_categorical(
            "gradient_clip_max_norm",
            [None, 1.0, 5.0, 10.0],
        ),
    )


def _sample_dqn_architecture_config(
    trial: optuna.Trial,
    config: ExperimentConfig,
) -> None:
    dqn = config.agent.dqn
    _sample_network_architecture(trial, dqn)
    dqn.hidden_dim = dqn.hidden_layers[0]


def _sample_double_dqn_architecture_config(
    trial: optuna.Trial,
    config: ExperimentConfig,
) -> None:
    double_dqn = config.agent.double_dqn
    _sample_network_architecture(trial, double_dqn)
    double_dqn.hidden_dim = double_dqn.hidden_layers[0]


def _sample_reinforce_architecture_config(
    trial: optuna.Trial,
    config: ExperimentConfig,
) -> None:
    reinforce = config.agent.reinforce
    _sample_network_architecture(trial, reinforce)
    reinforce.hidden_dim = reinforce.hidden_layers[0]
    # Widened to match _sample_reinforce_hyperparameter_config (see note
    # there). Kept identical across both modes so a batch_episodes value
    # found to work well in one search remains in-range in the other.
    reinforce.batch_episodes = trial.suggest_categorical(
        "batch_episodes",
        [4, 8, 16, 32, 64],
    )


def _sample_epsilon_schedule(
    trial: optuna.Trial,
    dqn: DQNAgentConfig | DoubleDQNAgentConfig,
) -> None:
    dqn.epsilon_schedule = trial.suggest_categorical(
        "epsilon_schedule",
        ["linear", "exponential", "cosine"],
    )
    dqn.epsilon_end = 0.01
    if dqn.epsilon_schedule == "linear":
        dqn.epsilon_decay_episodes = trial.suggest_categorical(
            "epsilon_decay_steps",
            [5_000, 10_000, 20_000, 50_000],
        )
    elif dqn.epsilon_schedule == "exponential":
        dqn.epsilon_decay = trial.suggest_categorical(
            "epsilon_decay",
            [0.999, 0.9995, 0.9999],
        )
    else:
        dqn.epsilon_decay_episodes = trial.suggest_categorical(
            "epsilon_decay_steps",
            [5_000, 10_000, 20_000, 50_000],
        )


def _sample_target_update(
    trial: optuna.Trial,
    dqn: DQNAgentConfig | DoubleDQNAgentConfig,
) -> None:
    dqn.target_update_type = trial.suggest_categorical(
        "target_update_type",
        ["hard", "soft"],
    )
    if dqn.target_update_type == "hard":
        dqn.target_update_frequency = trial.suggest_categorical(
            "target_update_frequency",
            [500, 1_000, 2_000, 5_000],
        )
    else:
        dqn.tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01])


def _sample_lr_scheduler(
    trial: optuna.Trial,
    dqn: DQNAgentConfig | DoubleDQNAgentConfig,
) -> None:
    scheduler = cast(
        str | None,
        trial.suggest_categorical(
            "lr_scheduler",
            [None, "step", "plateau"],
        ),
    )
    dqn.lr_scheduler = scheduler
    if scheduler == "step":
        dqn.step_lr_step_size = 100
        dqn.step_lr_gamma = 0.9
    elif scheduler == "plateau":
        dqn.reduce_lr_patience = 20


def _sample_network_architecture(
    trial: optuna.Trial,
    network_config: NetworkArchitectureConfig,
) -> None:
    family = trial.suggest_categorical("architecture_family", ["width", "depth"])
    if family == "width":
        width = trial.suggest_categorical("width", [32, 64, 128, 256, 512])
        network_config.hidden_layers = [width, width]
    else:
        depth = trial.suggest_categorical(
            "depth",
            ["1x256", "2x128", "3x64", "4x32"],
        )
        network_config.hidden_layers = _depth_layers(str(depth))

    network_config.activation = trial.suggest_categorical(
        "activation",
        ["relu", "leaky_relu", "elu", "selu", "tanh"],
    )
    network_config.weight_init = trial.suggest_categorical(
        "weight_init",
        ["he", "xavier", "orthogonal"],
    )
    norm_variant = trial.suggest_categorical(
        "normalization_variant",
        ["none", "bn_before", "bn_after", "layer", "dropout"],
    )
    _apply_network_normalization_variant(network_config, str(norm_variant))


def _apply_network_normalization_variant(
    network_config: NetworkArchitectureConfig,
    variant: str,
) -> None:
    network_config.normalization = None
    network_config.dropout = 0.0
    if variant == "bn_before":
        network_config.normalization = "batch"
        network_config.normalization_position = "before_activation"
    elif variant == "bn_after":
        network_config.normalization = "batch"
        network_config.normalization_position = "after_activation"
    elif variant == "layer":
        network_config.normalization = "layer"
    elif variant == "dropout":
        network_config.dropout = 0.1


def _depth_layers(depth: str) -> list[int]:
    mapping = {
        "1x256": [256],
        "2x128": [128, 128],
        "3x64": [64, 64, 64],
        "4x32": [32, 32, 32, 32],
    }
    return mapping[depth]


def _log_hidden_activation_heatmap(
    agent: ArchitectureDebugAgent,
    trial: optuna.Trial,
) -> None:
    observations = np.array(
        [
            [-1.0, 1.0, -0.5, 0.5, -0.2, 0.2, 0.0, 0.0],
            [0.0, 1.2, 0.0, -0.4, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.8, 0.2, -0.2, 0.1, -0.1, 1.0, 0.0],
            [0.0, 0.1, 0.0, -0.1, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    activations = agent.hidden_activation_snapshot(observations)
    fig, ax = plt.subplots(figsize=(8, 3))
    image = ax.imshow(activations, aspect="auto", cmap="viridis")
    ax.set_xlabel("hidden_unit")
    ax.set_ylabel("state")
    ax.set_title(f"Hidden activations trial {trial.number}")
    fig.colorbar(image, ax=ax)
    if wandb.run is not None:
        wandb.log({"architecture/hidden_activation_heatmap": wandb.Image(fig)})
    plt.close(fig)


def log_optuna_summary(
    study: optuna.Study,
    *,
    output_dir: str | Path = "visualisation",
) -> None:
    completed = [t for t in study.trials if t.value is not None]
    if len(completed) < 2:
        print("optuna_summary: need at least 2 completed trials, skipping.")
        return

    data_dir = Path(output_dir) / "data"
    plots_dir = Path(output_dir) / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    trials_df = study.trials_dataframe()
    csv_path = data_dir / f"{study.study_name}_trials.csv"
    trials_df.to_csv(csv_path, index=False)
    print(f"optuna_summary: wrote {csv_path}")

    importances = _safe_param_importances(study)
    if importances:
        _save_importance_bar_chart(importances, study.study_name, plots_dir)

    plot_specs: dict[str, plotly.graph_objects.Figure] = {
        "optimization_history": optuna.visualization.plot_optimization_history(study),
        "param_importances": optuna.visualization.plot_param_importances(study),
        "parallel_coordinate": optuna.visualization.plot_parallel_coordinate(study),
        "slice": optuna.visualization.plot_slice(study),
        "edf": optuna.visualization.plot_edf(study),
    }

    if importances and len(importances) >= 2:
        top_two = [
            name
            for name, _ in sorted(
                importances.items(), key=lambda kv: kv[1], reverse=True
            )[:2]
        ]
        try:
            plot_specs[f"contour_{top_two[0]}_vs_{top_two[1]}"] = (
                optuna.visualization.plot_contour(study, params=top_two)
            )
        except ValueError:
            pass

    if {"learning_rate", "gamma"}.issubset(_all_param_names(study)):
        try:
            plot_specs["contour_learning_rate_vs_gamma"] = (
                optuna.visualization.plot_contour(
                    study, params=["learning_rate", "gamma"]
                )
            )
        except ValueError:
            pass

    for name, figure in plot_specs.items():
        html_path = plots_dir / f"{study.study_name}_{name}.html"
        figure.write_html(str(html_path))
        if wandb.run is not None:
            try:
                wandb.log({f"optuna/{name}": wandb.Plotly(figure)})
            except Exception as exc:
                print(f"optuna_summary: could not log {name} to wandb: {exc}")


def _all_param_names(study: optuna.Study) -> set[str]:
    names: set[str] = set()
    for trial in study.trials:
        names.update(trial.params.keys())
    return names


def _safe_param_importances(study: optuna.Study) -> dict[str, float]:
    try:
        return optuna.importance.get_param_importances(study)
    except (RuntimeError, ValueError) as exc:
        print(f"optuna_summary: fANOVA importance failed ({exc}), falling back to MDI.")
        try:
            evaluator = optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
            return optuna.importance.get_param_importances(study, evaluator=evaluator)
        except Exception as exc2:
            print(f"optuna_summary: MDI importance also failed ({exc2}).")
            return {}


def _save_importance_bar_chart(
    importances: dict[str, float], study_name: str, plots_dir: Path
) -> None:
    sorted_items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(names))))
    ax.barh(names[::-1], values[::-1], color="#2563EB")
    ax.set_xlabel("importance (fraction of variance explained)")
    ax.set_title(f"Hyperparameter importance - {study_name}")
    fig.tight_layout()

    png_path = plots_dir / f"{study_name}_param_importance.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")

    if wandb.run is not None:
        table = wandb.Table(
            columns=["parameter", "importance"],
            data=[list(item) for item in sorted_items],
        )
        wandb.log(
            {
                "optuna/param_importance_table": table,
                "optuna/param_importance": wandb.Image(fig),
            }
        )

    plt.close(fig)


def _log_reward_distribution(rewards: list[float], trial: optuna.Trial) -> None:
    if wandb.run is None or len(rewards) < 1:
        return

    last_10 = rewards[-10:] if len(rewards) >= 10 else rewards

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(last_10, bins=8, color="#3b82f6", edgecolor="black", alpha=0.7)
    ax.set_xlabel("reward")
    ax.set_ylabel("frequency")
    ax.set_title(f"Reward distribution (last {len(last_10)} episodes)")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    wandb.log({"trial/reward_distribution": wandb.Image(fig)})
    plt.close(fig)


def _log_smoothed_learning_curve(rewards: list[float], trial: optuna.Trial) -> None:
    if wandb.run is None or len(rewards) < 1:
        return

    ema_values = []
    alpha = 0.1
    ema = rewards[0]
    ema_values.append(ema)

    for reward in rewards[1:]:
        ema = alpha * reward + (1 - alpha) * ema
        ema_values.append(ema)

    fig, ax = plt.subplots(figsize=(7, 4))
    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, alpha=0.3, color="#9ca3af", label="raw")
    ax.plot(episodes, ema_values, color="#ef4444", linewidth=2, label="EMA (alpha=0.1)")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title("Learning curve (smoothed)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    wandb.log({"trial/learning_curve_smoothed": wandb.Image(fig)})
    plt.close(fig)


def _log_architecture_params_text(
    config: ExperimentConfig, trial: optuna.Trial
) -> None:
    if wandb.run is None:
        return

    agent_cfg = config.agent
    if agent_cfg.name == "reinforce":
        cfg = agent_cfg.reinforce
        params_text = f"""
REINFORCE Architecture:
- hidden_layers: {cfg.hidden_layers}
- activation: {cfg.activation}
- weight_init: {cfg.weight_init}
- normalization: {cfg.normalization} ({cfg.normalization_position})
- dropout: {cfg.dropout}
- batch_episodes: {cfg.batch_episodes}
- learning_rate: {cfg.learning_rate}
- gamma: {cfg.gamma}
- gradient_clip_max_norm: {cfg.gradient_clip_max_norm}
"""
    else:
        agent_label = "Double DQN" if agent_cfg.name == "double_dqn" else "DQN"
        cfg = agent_cfg.double_dqn if agent_cfg.name == "double_dqn" else agent_cfg.dqn
        params_text = f"""
{agent_label} Architecture & Hyperparameters:
- hidden_layers: {cfg.hidden_layers}
- activation: {cfg.activation}
- weight_init: {cfg.weight_init}
- normalization: {cfg.normalization} ({cfg.normalization_position})
- dropout: {cfg.dropout}
- batch_size: {cfg.batch_size}
- learning_rate: {cfg.learning_rate}
- gamma: {cfg.gamma}
- epsilon_schedule: {cfg.epsilon_schedule}
- target_update_type: {cfg.target_update_type}
- lr_scheduler: {cfg.lr_scheduler}
"""

    wandb.log({"trial/architecture_params": wandb.Html(f"<pre>{params_text}</pre>")})


def _log_epsilon_decay_params(config: ExperimentConfig, trial: optuna.Trial) -> None:
    if wandb.run is None:
        return

    dqn = (
        config.agent.double_dqn
        if config.agent.name == "double_dqn"
        else config.agent.dqn
    )

    n_episodes = 1000
    episodes = np.arange(n_episodes)
    epsilon_vals = []

    for ep in episodes:
        if dqn.epsilon_schedule == "linear":
            frac = ep / dqn.epsilon_decay_episodes
            eps = max(
                dqn.epsilon_end,
                dqn.epsilon_start - (dqn.epsilon_start - dqn.epsilon_end) * frac,
            )
        elif dqn.epsilon_schedule == "exponential":
            eps = max(dqn.epsilon_end, dqn.epsilon_start * (dqn.epsilon_decay**ep))
        else:
            frac = ep / dqn.epsilon_decay_episodes
            eps = dqn.epsilon_end + 0.5 * (dqn.epsilon_start - dqn.epsilon_end) * (
                1 + np.cos(np.pi * frac)
            )
        epsilon_vals.append(eps)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(episodes, epsilon_vals, linewidth=2, color="#10b981")
    ax.set_xlabel("episode")
    ax.set_ylabel("epsilon")
    ax.set_title(f"Epsilon decay ({dqn.epsilon_schedule})")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    wandb.log({"trial/epsilon_decay_schedule": wandb.Image(fig)})
    plt.close(fig)


def _log_trial_ranking_table(study: optuna.Study) -> None:
    if wandb.run is None:
        return

    completed_trials = [t for t in study.trials if t.value is not None]
    if not completed_trials:
        return

    sorted_trials = sorted(
        completed_trials, key=lambda t: cast(float, t.value), reverse=True
    )[:10]

    rows = []
    for rank, trial in enumerate(sorted_trials, 1):
        param_count = trial.user_attrs.get("parameter_count", "N/A")
        key_params = {
            k: v
            for k, v in trial.params.items()
            if k
            in {
                "learning_rate",
                "gamma",
                "batch_size",
                "hidden_dim",
                "activation",
                "width",
                "depth",
                "batch_episodes",
            }
        }
        param_str = "; ".join(f"{k}={v}" for k, v in key_params.items())

        rows.append([rank, f"{trial.value:.2f}", str(param_count), param_str])

    table = wandb.Table(
        columns=["rank", "reward", "param_count", "key_params"], data=rows
    )
    wandb.log({"optuna/top_10_trials": table})


def _log_reward_vs_trial_number(study: optuna.Study) -> None:
    if wandb.run is None:
        return

    trial_nums = []
    rewards = []

    for trial in study.trials:
        if trial.value is not None:
            trial_nums.append(trial.number)
            rewards.append(trial.value)

    if not trial_nums:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(trial_nums, rewards, alpha=0.5, s=50, color="#6366f1")

    best_so_far = []
    current_best = float("-inf")
    for reward in rewards:
        if reward > current_best:
            current_best = reward
        best_so_far.append(current_best)
    ax.plot(trial_nums, best_so_far, color="#f59e0b", linewidth=2, label="best so far")

    ax.set_xlabel("trial number")
    ax.set_ylabel("avg_reward")
    ax.set_title("Optimization progress")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    wandb.log({"optuna/reward_vs_trial": wandb.Image(fig)})
    plt.close(fig)
