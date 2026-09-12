# Sprint long-horizon straight example

Continuation of [`../sprint_head_upright/`](../sprint_head_upright/). Same
upright head and alternating gait, with a closed-loop spawn-heading command
so the bird holds the spawn ray instead of drifting ~42° left.

Priority for this checkpoint: keep posture, then run straight, for as long as
possible. Speed is secondary.

## Battery vs the head-upright example

| Metric | Head-upright `14500` (10 s) | This `19297` (300 s, seed 123) |
|---|---|---|
| Survival | 97.9% | **100%** |
| Mean \|heading\| | 42.4° | **1.71°** |
| Straight-pose horizon | ~1 s (curves off the ray) | **294.9 s** (p10 294.9) |
| Straight-pose pass | 0% | **95.3%** |
| Spawn-forward distance | negative (ran off-line) | **247.9 m** |
| Head inverted | 0.6% | **0** |
| Thigh sign-flip | 0.48 | **0.48** |

600 s dual-seed check: survival 95.3% / 100%, mean heading 3.05° / 2.77°,
straight-pose horizon 561 s / 543 s.

Raw numbers: [`metrics.json`](metrics.json). Tracking clip:
[`play_video.mp4`](play_video.mp4) (30 s, 2.2 m/s command, zero yaw range;
the owner still writes a heading/cross-track yaw command into the existing
61D observation).

## What changed

Three stacked fixes, none of which change the 61D actor / 14D action contract:

1. **Steer again.** Forward-only training had forgotten bidirectional yaw.
   `microduck_sprint_steerfix_flat` restores `tracking_ang_vel` before the
   closed loop is turned on.
2. **Capture the real spawn yaw.** Reset events commit after command resample,
   so spawn heading is read in `_update_command`, not `_resample_command`.
3. **Correct lateral drift.** Heading-only feedback held yaw but walked
   sideways. `spawn_cross_track_stiffness: 0.08` blends cross-track error into
   the same yaw-command channel.

## Train

Resume the head-upright example, restore yaw tracking, then close the spawn
loop:

```bash
uv run --no-sync scripts/train_sprint_head_repair.py \
  --task microduck_sprint_steerfix_flat \
  --load-run "$(pwd)/examples/sprint_head_upright" \
  --checkpoint 14500 \
  --iterations 1500 --save-interval 100 \
  --algo-log-name rsl_rl_ppo_steerfix

uv run --no-sync scripts/train_sprint_head_repair.py \
  --task microduck_sprint_straightfix_flat \
  --load-run <steerfix-run> \
  --checkpoint <last-iter> \
  --iterations 8000 --save-interval 200 \
  --algo-log-name rsl_rl_ppo_longstraight
```

## Eval / play

Use the straightfix owner. The default `microduck_sprint_flat` owner does not
hold spawn heading, so a 10 s battery on that owner will not show this loop.

```bash
uv run --no-sync scripts/eval_sprint_speed.py \
  examples/sprint_straight_long/model_19297.pt \
  --owner microduck_sprint_straightfix_flat/mujoco \
  --duration 300 --num-envs 128

uv run --no-sync scripts/diagnose_gait.py \
  examples/sprint_straight_long/model_19297.pt --speed 2.2 \
  --owner microduck_sprint_straightfix_flat/mujoco \
  --hold-spawn-heading --heading-stiffness 2.5 \
  --heading-yaw-limit 0.35 --cross-track-stiffness 0.08

MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa uv run microduck-eval --algo ppo \
  --task microduck_sprint_straightfix_flat --sim mujoco \
  --load-run "$(pwd)/examples/sprint_straight_long" \
  algo.checkpoint=19297 algo.num_envs=1 training.play_steps=1500 \
  training.cam_tracking=true training.cam_tracking_extra_envs=0 \
  training.cam_azimuth=135 training.cam_elevation=-15 \
  'env.commands.twist.ranges.lin_vel_x=[2.2,2.2]' \
  'env.commands.twist.ranges.lin_vel_y=[0.0,0.0]' \
  'env.commands.twist.ranges.ang_vel_z=[0.0,0.0]'
```
