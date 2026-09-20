# microduck_rl_unilab

[English](README.md) | [中文](README.zh.md)

Pollen Robotics MicroDuck RL training on **UniLab's package distribution only**
(no UniLab source checkout). Tasks reproduce upstream
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)
@ `29e887e` (alignment contract:
`src/microduck_rl_unilab/tasks/microduck/alignment_contract.py`;
audit script: `scripts/audit_microduck_alignment.py`).

This repository shows the two public UniLab integration seams:

1. **Task registration**: `UNILAB_EXTRA_REGISTRY_PACKAGES` lets UniLab's
   `ensure_registries()` import this repo's task package (including spawn
   subprocesses).
2. **Config composition**: Hydra `--config-dir` appends
   `src/microduck_rl_unilab/conf/<algo>` to the matching UniLab training
   script's config search path so external `task=<task>/<sim>` owner YAML
   participates in composition. The conf tree is split by algorithm
   (`conf/ppo/`, `conf/sac/`), matching UniLab's in-repo layout.

Train, eval, and runner/learner/collector all come from the `unilab` wheel.
This repo only ships task code (manager terms / recovery terms / BAM action,
and so on), owner configs, and robot XML assets.

## Task × algorithm × backend matrix

| task | ppo | sac |
|------|-----|-----|
| `microduck_velocity_flat` | mujoco, mjwarp | mujoco, mjwarp |
| `microduck_sprint_flat` | mujoco | — |
| `microduck_sprint_robust_flat` | mujoco | — |
| `microduck_sprint_gaitfix_flat` | mujoco | — |
| `microduck_sprint_rollfix_flat` | mujoco | — |
| `microduck_velstand_flat` | mujoco | — |
| `microduck_standup_flat` | mujoco | — |
| `microduck_ground_pick_flat` | mjwarp | — |
| `microduck_sitstand_flat` | mjwarp | — |

Tasks stay 1:1 with upstream microduck_rl: BAM (bam xl330-m6 voltage servo,
`BamVoltageAction` per-substep torque path) is the only upstream actuator
model, so every task here uses BAM. Early simplified position-actuated
variants were dropped. BAM's substep state-feedback contract
(`SimBackend.set_pre_step_control`) is available on both mujoco and mjwarp
(the latter since unisim PR #20).

## Install

```bash
git clone https://github.com/unilabsim/microduck_rl_unilab.git
cd microduck_rl_unilab
uv sync
```

> **Dependencies**: `unilab[mujoco,mjwarp]==1.0.0`, `unilab-rl==1.0.0`, and
> `unisim-core==1.0.0` all resolve from production PyPI. unilab 1.0.0 includes
> in-tree microduck task removal (PR #1495), the `read_reset_root_pose` base
> API (PR #1494), and the `unilab-rl==1.0.0` bump (PR #1496). unisim-core
> 1.0.0 includes mjwarp `set_pre_step_control` (PR #20, required for BAM ×
> mjwarp).

## Train

Run from the repository root (`env.scene.model_file` is resolved relative to
the repo root):

```bash
uv run microduck-train --algo ppo --task microduck_velocity_flat --sim mujoco
uv run microduck-train --algo ppo --task microduck_sprint_flat --sim mujoco
uv run microduck-train --algo ppo --task microduck_velocity_flat --sim mjwarp
uv run microduck-train --algo ppo --task microduck_standup_flat --sim mujoco
uv run microduck-train --algo sac --task microduck_velocity_flat --sim mujoco
```

Short smoke (4 envs, 2 iterations, no play):

```bash
uv run microduck-train --algo ppo --task microduck_velocity_flat --sim mujoco \
  algo.num_envs=4 algo.max_iterations=2 training.no_play=true training.play_env_num=4
```

Any UniLab Hydra override is passed through (for example `algo.num_envs=512`,
`training.logger=wandb`).

## Eval

```bash
uv run microduck-eval --algo ppo --task microduck_velocity_flat --sim mujoco --load-run -1
```

Sprint eval protocol: 2.20 m/s forward command, 1 s warmup, 10 s measurement:

```bash
uv run --no-sync scripts/eval_sprint_speed.py \
  examples/sprint_speed_1p68/model_11997.pt
```

Three-stage robustify from a trained speed policy (pin the 1.65–2.20 m/s
command band, then add push / CoM / tilt):

```bash
uv run --no-sync scripts/train_sprint_robust.py \
  --load-run <sprint-run> --checkpoint <last-iter> --stages B
```

## Example checkpoints

| Example | Speed | Survival | Head inverted | Notes |
|---|---|---|---|---|
| [`examples/sprint_speed_1p68/`](examples/sprint_speed_1p68/) | 1.682 m/s | 89.8% | ~91% | Speed recipe. Thighs already alternate; neck is folded and the head is inverted most of the time. |
| [`examples/sprint_head_upright/`](examples/sprint_head_upright/) | 1.315 m/s | 97.9% | 0.6% | Same alternating gait with the crown up and face forward; 10 s heading still ~42° left. |
| [`examples/sprint_straight_long/`](examples/sprint_straight_long/) | ~0.83 m/s long-horizon | 300 s 100% | 0 | Continued training with closed-loop heading + lateral error. 300 s mean heading 1.7°, straight window 295 s. |

Each directory has `model_*.pt`, visualizations, `metrics.json`, and
reproduce commands. `model.pt` is about 4.7 MB and is stored in git.

Long-horizon straight eval (needs the closed-loop owner, not default
`microduck_sprint_flat`):

```bash
uv run --no-sync scripts/eval_sprint_speed.py \
  examples/sprint_straight_long/model_19297.pt \
  --owner microduck_sprint_straightfix_flat/mujoco \
  --duration 300 --num-envs 128
```

## Tests

```bash
uv run pytest tests/ -x -q
```

`tests/conftest.py` recreates both seams: it sets
`UNILAB_EXTRA_REGISTRY_PACKAGES` and uses a Hydra `SearchPathPlugin` to append
this repo's `conf/<algo>` to the config search path (tests do not go through
the CLI, so they cannot use `--config-dir`).

## Asset policy

All MicroDuck assets (7 XML + 47 STL + upstream LICENSE + sha256 manifest)
are in git and work out of the box. `assets.py` is a fallback only: missing
files are filled from the Hugging Face dataset
[`unilabsim/unilab-robots`](https://huggingface.co/datasets/unilabsim/unilab-robots)
(cold path). STLs come from upstream pollen-robotics/microduck_rl
(Apache-2.0, see `assets/robots/microduck/LICENSE.pollen-robotics.txt`).
Integrity is checked by `assets/robots/microduck/assets.sha256` and
`tests/test_asset_contract.py`.

## Layout

```text
assets/robots/microduck/            # robot XML + STL + LICENSE + sha256 manifest (all in git)
src/microduck_rl_unilab/
├── assets.py                       # cold-path asset materialization (snapshot_download fallback)
├── cli.py                          # microduck-train / microduck-eval: env var + --config-dir injection
├── conf_searchpath.py              # Hydra SearchPathPlugin (programmatic compose for tests/scripts)
├── conf/
│   ├── ppo/task/microduck_{velocity_flat,sprint_flat,sprint_robust_flat,velstand_flat,standup_flat,ground_pick_flat,sitstand_flat}/
│   └── sac/task/microduck_velocity_flat/
└── tasks/
    ├── __init__.py                 # __unilab_registry_modules__
    └── microduck/
        ├── __init__.py             # task registry.register_env
        ├── manager_terms.py        # velocity command / reward terms
        ├── sprint_terms.py         # sprint speed/heading rewards and speed curriculum
        ├── recovery_terms.py       # velstand fall-recovery terms
        ├── standup_terms.py        # standup / ground_pick / sitstand terms
        ├── bam_action.py           # BAM voltage-servo action term
        ├── deploy_contract.py      # obs/action dimension contract
        ├── alignment_contract.py   # upstream microduck_rl @ 29e887e alignment table
        └── sim2real_notes.py       # sim2real notes
tests/                              # contract/alignment suite migrated from UniLab
scripts/                            # alignment audit / compare / rollout scripts
```
