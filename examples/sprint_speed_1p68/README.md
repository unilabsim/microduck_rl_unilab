# Sprint speed example (1.68 m/s)

This is the published **forward-speed** checkpoint for `MicroduckSprintFlat`.
The battery is a 2.20 m/s command, 1 s warmup, 10 s measurement, 512 environments.

## Battery

| Metric | Value |
|---|---|
| Mean body-forward speed | **1.682 m/s** |
| Survival | 89.8% |
| Heading error | 42.4° |
| Head inverted time | **91%** (known limit) |
| Head-up tilt | 127° |
| Thigh split sign-flip | 0.47 (alternating) |
| Foot scissor peak-to-peak | 0.24 m |

Raw numbers: [`metrics.json`](metrics.json). Side-view loop: [`play_video_side.gif`](play_video_side.gif).

This checkpoint is **not** a posture example. The thighs already trade places
(that is why it is fast). Balance comes from folding `neck_pitch` onto its
stop, so the head spends most of the run upside down. The follow-up example
in `examples/sprint_head_upright/` keeps the same alternating gait and puts
the head back upright, at about 1.31 m/s.

Robustify stages that produced this checkpoint (same 2.2 m/s battery):

| Stage | Checkpoint | Speed | Survival | Heading |
|---|---|---|---|---|
| A | `model_11848` | 1.672 m/s | 83.6% | 44.7° |
| **B (published)** | **`model_11997`** | **1.682 m/s** | **89.8%** | **42.4°** |
| C | `model_12196` | 1.676 m/s | 81.6% | 35.2° |

Stage C added more push/CoM/tilt and dropped survival, so Stage B is the example.

## Train

Speed discovery, then Stage B robustify (push 0.06, trunk CoM 5 mm, 1.5° tilt):

```bash
uv run microduck-train --algo ppo --task microduck_sprint_flat --sim mujoco \
  algo.num_envs=2048 training.no_play=true

uv run --no-sync scripts/train_sprint_robust.py \
  --load-run <speed-run> --checkpoint <last-iter> --stages B
```

The owner that produced `model_11997.pt` is `microduck_sprint_robust_flat`,
with the command band frozen at 1.65–2.20 m/s.

## Eval / play

```bash
uv run --no-sync scripts/eval_sprint_speed.py \
  examples/sprint_speed_1p68/model_11997.pt

MUJOCO_GL=osmesa uv run microduck-eval --algo ppo \
  --task microduck_sprint_robust_flat --sim mujoco \
  --load-run "$(pwd)/examples/sprint_speed_1p68" \
  algo.checkpoint=11997 \
  training.play_env_num=1 training.play_steps=600 \
  training.cam_tracking=true training.cam_azimuth=90 \
  env.events.push_robot=null \
  'env.commands.twist.ranges.lin_vel_x=[2.2,2.2]'
```

`algo.load_run` must be a directory that contains `model_11997.pt`.
