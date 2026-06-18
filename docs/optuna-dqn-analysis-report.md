# Optuna And DQN Analysis Report

## Scope

This report analyzes why:

- an Optuna-selected DQN configuration could report average reward near `200` during tuning
- the same hyperparameters could perform poorly when rerun later
- the general DQN baseline in this repository performed weakly

The analysis is based on:

- static review of the training, Optuna, evaluation, reward-shaping, and checkpoint code
- local runtime probes
- inspection of real project runs in Weights & Biases

## Status

This document now reflects two things:

1. the original failure modes found in the repository
2. the first round of fixes already applied on branch `fix/optuna-dqn-repro`

When a finding below describes the repository in the past tense, it refers to the behavior before the fixes on this branch.

## Executive Summary

The Optuna-to-rerun mismatch was not caused by one bug. It came from several interacting problems:

1. Optuna selected trials using late training reward instead of evaluation reward.
2. Training and evaluation could use different reward definitions.
3. The saved "best checkpoint" was a fresh retrain instead of the actual winning trial artifact.
4. DQN runs were not reproducible because model initialization was not fully seeded.
5. The DQN update schedule was too weak and used episode-based semantics where step-based semantics were needed.
6. Checkpoint naming and storage were too ambiguous.

The branch `fix/optuna-dqn-repro` addresses the highest-signal issues in that list.

## Original Findings

### 1. Optuna optimized the wrong objective

Originally, [`training/optuna_search.py`](/Users/jan/Desktop/Programowanie/pwr/semestr_5/nn-3/training/optuna_search.py:35) scored each trial using the last 10 training episodes rather than evaluation performance.

Implications:

- the optimized objective was training reward, not evaluation reward
- the score was based on a very small and noisy window
- late lucky episodes could dominate trial ranking
- if reward shaping was active, the optimized signal was shaped reward rather than raw environment reward

### 2. Training and evaluation reward definitions could diverge

Training could wrap the environment with [`ShapedLunarLander`](/Users/jan/Desktop/Programowanie/pwr/semestr_5/nn-3/rewards/shaping.py:58), while evaluation always used a fresh raw Gym environment in [`training/evaluate.py`](/Users/jan/Desktop/Programowanie/pwr/semestr_5/nn-3/training/evaluate.py:22).

Implications:

- training reward and evaluation reward were not necessarily comparable
- a trial could look strong during training while still evaluating poorly

### 3. Reward shaping behavior differed across code paths

The config schema default enabled reward shaping, while `optimize.py` used to silently override that behavior unless the CLI flag was passed.

Implications:

- `train.py` and `optimize.py` could run meaningfully different experiments
- "same hyperparameters" did not necessarily mean "same setup"

### 4. The saved Optuna "best checkpoint" was not the winning trial artifact

Originally, the post-study save path retrained from scratch using the best parameters and then saved that new model.

Implications:

- the exported checkpoint could be worse than the actual winning trial
- stochasticity could break reproducibility even when the hyperparameters matched

### 5. DQN reproducibility was broken

Originally, the agent seeded its local NumPy RNG but Torch initialization was not seeded before network construction.

Observed local probe before the fix:

- two DQN agents created with the same configured seed did not have identical initial weights
- direct probe result: `weights_equal=False`
- direct probe result: `max_abs_diff=1.7006360292434692`

Implications:

- same config and same seed still produced different initial networks
- Optuna could rank parameter sets partly on initialization luck

### 6. DQN update scheduling was too weak

Originally, the trainer collected an entire episode and then performed only one `agent.update()` call per episode.

Implications:

- replay data accumulated much faster than the network learned from it
- epsilon and target-network progress were tied to sparse update calls rather than actual interaction volume
- DQN learning quality was predictably poor

### 7. DQN scheduling semantics were episode-based instead of step-based

Originally, epsilon decay and target updates were driven by optimizer update count rather than environment step count.

Implications:

- tuned epsilon parameters were hard to interpret
- changing the update schedule implicitly changed exploration behavior

### 8. Checkpoint hygiene was poor

The checkpoint directory mixed artifacts from different agents and formats. During inspection, `checkpoints/best.pt` was a JSON artifact from a random agent run rather than a Torch DQN checkpoint.

Implications:

- "best model" references were ambiguous
- stale artifacts could be mistaken for current outputs

## Runtime Evidence

Before the fixes, I ran two short DQN probes locally with `wandb` disabled and otherwise the same config:

- Run 1: `last10_avg=-286.22`, `overall_avg=-196.47`, `best=42.63`
- Run 2: `last10_avg=-233.99`, `overall_avg=-187.00`, `best=69.68`

That confirmed same-config runs drifted materially.

After the seeding and scheduling fixes on this branch:

- same-seed DQN agents now initialize identically
- repeated short same-config runs now reproduce exactly

## W&B Confirmation

The train-vs-eval mismatch is visible in the real project runs, not just in local probes.

Examples from DQN architecture tuning:

- Run `r3diro6m` (`architecture-trial-23`)
  - `train/reward = 254.14`
  - `eval/avg_reward = -202.51`
- Run `i1s1iu6n` (`architecture-trial-22`)
  - `train/reward = 211.67`
  - `eval/avg_reward = -342.09`
- Run `idkn6ldw` (`architecture-trial-19`)
  - `train/reward = 233.98`
  - `eval/avg_reward = 79.71`

This confirms that the old Optuna objective was selecting on a noisy signal that often did not correspond to policy quality under evaluation.

Additional observation from W&B:

- the historical DQN Optuna runs recorded `reward_shaping.enabled = false`
- that matched the previous `optimize.py` override behavior rather than the config schema default

## Applied Fixes On This Branch

The branch `fix/optuna-dqn-repro` now includes:

- deterministic seeding for Python, NumPy, and Torch
- deterministic Torch seeding before model construction
- Optuna objective switched from last-10 training reward to final evaluation reward
- per-trial Optuna artifact saving so the winning model comes from the actual winning trial
- best-checkpoint export now copies the winning trial artifact instead of retraining
- explicit DQN reward-shaping settings in config files
- removal of the silent reward-shaping override default in `optimize.py`
- raw training reward logging alongside shaped reward where relevant
- step-based DQN exploration and target-update semantics
- updated DQN defaults and Optuna search spaces to match step-based scheduling

## Validation Of Applied Fixes

Validation completed on this branch:

- `uv run pyright` passes
- `uv run ruff check .` passes
- same-seed DQN agents now initialize with identical weights
- repeated short same-config DQN runs now reproduce exactly
- short DQN probe after the step-based epsilon fix ended with `final_epsilon=0.317974`, which is consistent with the new step-based schedule rather than collapsing immediately to `0.01`

## Remaining Recommendations

The current branch fixes the most immediate correctness and reproducibility problems. The next steps should be:

1. Rerun a plain DQN baseline on this branch and compare it to the old runs.
2. Rerun a smaller Optuna study using the new evaluation-based objective.
3. Evaluate top candidates across multiple seeds rather than a single seed.
4. Consider separating checkpoint directories per agent/study to eliminate artifact ambiguity entirely.
5. If DQN remains weak after the corrected pipeline, improve the algorithm itself rather than the experiment plumbing:
   - better warmup and update cadence
   - Double DQN
   - Huber loss
   - prioritized replay

## Conclusion

The original Optuna and DQN setup was not measuring, saving, or reproducing the thing you actually cared about: a strong and repeatable DQN policy on LunarLander. The main failures were experimental hygiene and training semantics, not a mysterious Optuna bug. The fixes on `fix/optuna-dqn-repro` correct the most important parts of that pipeline, and they should make the next round of results substantially more trustworthy.
