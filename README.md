# RL Lunar Lander

This repository contains a reinforcement learning project scaffold for `LunarLander-v3` built on top of `gymnasium`, `torch`, and `omegaconf`.

The project is intentionally in an early but structured state:

- there is a working training entrypoint
- there is a working random baseline agent
- there is a placeholder `REINFORCE` agent wired into the same training stack
- training is organized so future algorithms can reuse rollout, evaluation, checkpointing, and config loading

## Repository Layout

High-level structure:

```text
agents/
configs/
models/
notebooks/
rewards/
training/
utils/
visualisation/
train.py
pyproject.toml
```

Important files and folders:

- `train.py`
  Main CLI entrypoint for training.

- `agents/`
  Agent implementations and agent construction logic.

  Current contents:
  - `baseline_agent.py`: random discrete-action baseline
  - `reinforce_agent.py`: placeholder REINFORCE agent scaffold
  - `factory.py`: builds the agent from config
  - `protocol.py`: common interface expected by the trainer

- `configs/`
  Experiment YAML files.

  Current contents:
  - `baseline.yaml`: main baseline training config
  - `baseline_logging.yaml`: longer-running baseline preset
  - `agents/random.yaml`: random agent overlay
  - `agents/reinforce.yaml`: reinforce agent overlay

- `models/`
  Torch model definitions used by learning-based agents.

  Current contents:
  - `policy_network.py`: simple MLP policy network

- `training/`
  Shared training infrastructure.

  Current contents:
  - `config.py`: structured config schema and YAML loading
  - `trainer.py`: top-level orchestration loop
  - `rollout.py`: environment interaction and episode collection
  - `evaluate.py`: evaluation helpers and console logging
  - `checkpoint.py`: checkpoint scheduling policy
  - `buffers.py`: transition and trajectory storage

- `utils/checkpointing.py`
  Shared Torch checkpoint save/load utility with compatibility helpers.

- `rewards/original.py`
  Placeholder for reward-related experimentation.

- `notebooks/`
  Jupyter notebooks for exploration and experimentation.

- `checkpoints/`
  Saved checkpoints from previous runs. This directory already contains generated artifacts.

- `visualisation/`
  Reserved for plots and result visualization assets. The folder exists, but there is no committed plotting script in the current state of the repository.

## Training Architecture

The training flow is intentionally split into modules rather than kept in one monolithic script.

Current flow:

1. `train.py` parses CLI arguments and loads the experiment config.
2. `training/config.py` loads the root YAML config and merges the selected agent overlay.
3. `training/trainer.py` creates the Gym environment and the configured agent.
4. `training/rollout.py` collects one episode worth of transitions.
5. The agent receives transitions through `observe(...)`.
6. The trainer calls `agent.update()` according to `training.update_every`.
7. `training/evaluate.py` optionally runs evaluation rollouts.
8. `training/checkpoint.py` decides when to save checkpoints.

The agent protocol currently expects:

- `act(observation, explore=True)`
- `observe(transition)`
- `update()`
- `reset()`
- `save(path)`
- `load(path)`

That interface is what future algorithms should implement.

## Agents

### Random Agent

File: `agents/baseline_agent.py`

This is the only fully working baseline right now. It:

- assumes a discrete action space
- samples random actions
- supports the trainer protocol
- saves a lightweight JSON checkpoint

Use it when:

- you want a sanity check that environment wiring works
- you want to validate training loops, evaluation loops, and checkpoint scheduling
- you want a reference baseline before implementing actual learning

### Reinforce Agent

File: `agents/reinforce_agent.py`

This is a scaffold, not a real REINFORCE implementation yet.

## Configuration System

Configs use OmegaConf structured dataclasses from `training/config.py`.

There are two layers:

- a root experiment config such as `configs/baseline.yaml`
- an agent overlay such as `configs/agents/random.yaml`

The root config selects the overlay via:

```yaml
agent_config: agents/random.yaml
```

Current top-level config sections:

- `env`
- `training`
- `evaluation`
- `checkpoint`
- `agent_config`

The agent overlay defines:

- `agent.name`
- `agent.seed`
- agent-specific subconfig such as `agent.reinforce`

### Baseline Configs

`configs/baseline.yaml` is the default config used by:

```bash
uv run python train.py --config configs/baseline.yaml
```

`configs/baseline_logging.yaml` is a longer-running preset intended for bigger runs. It currently differs mainly by:

- more episodes
- less frequent console logging
- evaluation enabled

## Checkpointing

There are two checkpoint paths in the current codebase:

- `RandomAgent` writes a simple JSON file through its own `save()` implementation
- `ReinforceAgent` uses `utils/checkpointing.py` and writes Torch checkpoint dictionaries

`training/checkpoint.py` controls when checkpoints are saved. It currently supports:

- periodic saves every `checkpoint.frequency` episodes
- best-checkpoint saving based on highest episode reward

Important note:

- the checkpoint scheduler is generic
- the actual checkpoint format depends on the agent implementation

So if you add new agents, decide early whether they should use:

- Torch-style state dict checkpoints
- a simpler serialized format

For any model-based agent, prefer reusing `utils/checkpointing.py`.

## Evaluation

Evaluation is configured in the `evaluation` section of the root config.

Current behavior:

- evaluation is skipped entirely if `evaluation.enabled` is `false`
- otherwise it runs every `evaluation.frequency` episodes
- it uses `explore=False` when calling `agent.act(...)`

This is enough for baseline comparisons, but it is still simple. There is currently no:

- separate evaluation seed schedule config
- CSV/JSON logging of evaluation outputs
- plot generation

## Development Setup

This project uses `uv`.

### Install dependencies

If the virtual environment is not prepared yet:

```bash
uv sync
```

If you only want to run the project commands and let `uv` resolve them on demand, the existing setup may already be enough.

### Run training

Default baseline:

```bash
uv run python train.py --config configs/baseline.yaml
```

Longer random-baseline preset:

```bash
uv run python train.py --config configs/baseline_logging.yaml
```

Switch to the REINFORCE scaffold by changing the overlay in the root config or by creating another root config that points to:

```yaml
agent_config: agents/reinforce.yaml
```

### Lint

```bash
uv run ruff check .
```

### Type-check

```bash
uv run pyright
```

## How To Develop New Algorithms

The current codebase is set up so new algorithms should fit into the existing structure rather than branching into separate training scripts.

Recommended approach:

1. Add or extend an agent config dataclass in `training/config.py`.
2. Add a YAML overlay in `configs/agents/`.
3. Implement the agent in `agents/`.
4. Reuse `models/` for neural networks instead of defining them inline in the agent.
5. Reuse `training/buffers.py` for trajectories or replay storage. Extend it if needed.
6. Register the agent in `agents/factory.py`.
7. Keep `train.py` unchanged unless the CLI itself really needs a new option.

## Practical Conventions

When extending this repository, keep these rules:

- keep the training entrypoint in `train.py`
- keep config in YAML
- validate config through OmegaConf structured dataclasses
- avoid putting algorithm-specific logic into `training/trainer.py`
- keep trainer responsibilities generic: orchestration, not algorithm math
- keep checkpoint save/load logic inside the agent boundary where practical

## License

See `LICENSE`.
