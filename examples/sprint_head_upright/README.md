# Sprint head-upright example (1.31 m/s)

Continuation of [`../sprint_speed_1p68/`](../sprint_speed_1p68/). Same
alternating gait, with the head kept upright and facing forward.

Official battery: 2.20 m/s command, 1 s warmup, 10 s measurement, 512 environments.

## Battery vs the speed example

| Metric | Speed example `11997` | This checkpoint `14500` |
|---|---|---|
| Mean body-forward speed | 1.682 m/s | **1.315 m/s** |
| Survival | 89.8% | **97.9%** |
| Head inverted time | 91% | **0.6%** |
| Head-up tilt | 127° | **38.5°** |
| Head yaw | 134° | **1.9°** |
| Head roll | 25.4° (joint stop) | **1.4°** |
| Thigh split sign-flip | 0.47 | **0.48** |
| Foot scissor peak-to-peak | 0.24 m | **0.23 m** |
| Heading error | 42.4° | 42.4° |

Raw numbers: [`metrics.json`](metrics.json). Side-view loop: [`play_video_side.gif`](play_video_side.gif).

The gait guards held: the thighs still trade places. The cost is 0.37 m/s of
top speed. Heading drift is **not** fixed in this checkpoint; the closed-loop
straight-run continuation is [`../sprint_straight_long/`](../sprint_straight_long/).

## Train

Resume the speed example, ramp the head contract (`microduck_sprint_gaitfix_flat`),
then close the `head_roll` stop (`microduck_sprint_rollfix_flat`):

```bash
uv run --no-sync scripts/train_sprint_head_repair.py \
  --task microduck_sprint_gaitfix_flat \
  --load-run "$(pwd)/examples/sprint_speed_1p68" \
  --checkpoint 11997 \
  --iterations 5000 --save-interval 250 \
  --algo-log-name rsl_rl_ppo_headfix

uv run --no-sync scripts/train_sprint_head_repair.py \
  --task microduck_sprint_rollfix_flat \
  --load-run <gaitfix-run> \
  --checkpoint 13500 \
  --iterations 3000 --save-interval 250 \
  --algo-log-name rsl_rl_ppo_rollfix \
  --max-head-roll 15 --max-head-yaw 10
```

`reward_curriculum` keys off `env.common_step_counter`, which restarts on each
new run, so keep each owner as one continuous job.

## Eval / play

```bash
uv run --no-sync scripts/eval_sprint_speed.py \
  examples/sprint_head_upright/model_14500.pt

uv run --no-sync scripts/diagnose_gait.py \
  examples/sprint_head_upright/model_14500.pt --speed 2.2

MUJOCO_GL=osmesa uv run microduck-eval --algo ppo \
  --task microduck_sprint_rollfix_flat --sim mujoco \
  --load-run "$(pwd)/examples/sprint_head_upright" \
  algo.checkpoint=14500 \
  training.play_env_num=1 training.play_steps=600 \
  training.cam_tracking=true training.cam_azimuth=90 \
  env.events.push_robot=null \
  'env.commands.twist.ranges.lin_vel_x=[2.2,2.2]'
```
