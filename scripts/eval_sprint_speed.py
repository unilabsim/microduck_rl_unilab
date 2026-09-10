"""Evaluate a Sprint checkpoint with the published Running battery.

The primary comparison uses a 2.20 m/s command, one second of warm-up, and a
ten-second measurement window. Speed samples stop accumulating for an
environment after its trunk exceeds the configured fall angle.

Usage:
    uv run --no-sync scripts/eval_sprint_speed.py model_7499.pt
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.cli import package_root
from unilab.training import algo_config_dict
from unilab.utils.rotation import np_wrap_to_pi

from microduck_rl_unilab.conf_searchpath import register_conf_search_path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = package_root() / "conf" / "ppo"


def _compose_owner(
    *,
    speed: float,
    duration_s: float,
    seed: int,
    stress: bool = False,
) -> Any:
    register_conf_search_path()
    GlobalHydra.instance().clear()
    owner = (
        "microduck_sprint_robust_flat/mujoco" if stress else "microduck_sprint_flat/mujoco"
    )
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={owner}"])
    with open_dict(cfg):
        cfg.algo.seed = seed
        cfg.env.max_episode_seconds = duration_s + 1.0
        twist = cfg.env.commands.twist
        twist.ranges.lin_vel_x = [speed, speed]
        twist.ranges.lin_vel_y = [0.0, 0.0]
        twist.ranges.ang_vel_z = [0.0, 0.0]
        twist.rel_standing_envs = 0.0
        twist.rel_forward_envs = 0.0
        twist.turn_in_place_fraction = 0.0
        twist.resampling_time_range = [duration_s + 1.0, duration_s + 1.0]
        for name in cfg.env.curriculum:
            cfg.env.curriculum[name] = None
        # The battery reads kinematics directly. Keeping tilt/nan terminations
        # recycles unsafe rows, while disabling rewards prevents dead rows from
        # tripping finite checks before their reset is applied.
        for name in cfg.reward:
            cfg.reward[name] = None
        if stress:
            # Held-out stress from the playground release record: wider trunk
            # CoM than the last training stage, plus the stage-C push/tilt.
            tilt = math.radians(2.0)
            cfg.env.events.push_robot.params.velocity_range.x = [-0.10, 0.10]
            cfg.env.events.push_robot.params.velocity_range.y = [-0.10, 0.10]
            cfg.env.events.base_com.params.com_range.x = [-0.010, 0.010]
            cfg.env.events.base_com.params.com_range.y = [-0.010, 0.010]
            cfg.env.events.base_com.params.com_range.z = [-0.010, 0.010]
            cfg.env.events.head_com.params.com_range.x = [-0.006, 0.006]
            cfg.env.events.head_com.params.com_range.y = [-0.006, 0.006]
            cfg.env.events.head_com.params.com_range.z = [-0.006, 0.006]
            cfg.env.events.reset_base.params.pose_range.roll = [-tilt, tilt]
            cfg.env.events.reset_base.params.pose_range.pitch = [-tilt, tilt]
    return cfg


def _quantile(values: np.ndarray, quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if values.size else None


def evaluate(
    checkpoint: Path,
    *,
    speed: float = 2.2,
    num_envs: int = 512,
    duration_s: float = 10.0,
    warmup_s: float = 1.0,
    fall_tilt_deg: float = 70.0,
    seed: int = 123,
    device: str | None = None,
    stress: bool = False,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if speed < 0.0:
        raise ValueError("speed must be nonnegative")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if not 0.0 <= warmup_s < duration_s:
        raise ValueError("warmup_s must be in [0, duration_s)")
    if not 0.0 < fall_tilt_deg <= 180.0:
        raise ValueError("fall_tilt_deg must be in (0, 180]")

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = _compose_owner(
        speed=speed,
        duration_s=warmup_s + duration_s,
        seed=seed,
        stress=stress,
    )
    importlib.import_module("microduck_rl_unilab.tasks.microduck")
    registry.ensure_registries()
    override = BackendAdapter(
        cfg,
        root_dir=REPO_ROOT,
        algo_name="ppo",
    ).build_task_env_cfg_override()
    raw_env = cast(
        Any,
        registry.make(
            "MicroduckSprintFlat",
            sim_backend="mujoco",
            num_envs=num_envs,
            env_cfg_override=override,
        ),
    )
    wrapped_env: RslRlVecEnvWrapper | None = None
    try:
        wrapped_env = RslRlVecEnvWrapper(raw_env, device=device)
        train_cfg = normalize_ppo_train_cfg(algo_config_dict(cfg))
        train_cfg.setdefault("runner", {})["logger"] = "none"
        train_cfg["logger"] = "none"
        algorithm_cfg = train_cfg.get("algorithm")
        if isinstance(algorithm_cfg, dict):
            algorithm_cfg["enable_compile"] = False
        runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=device)
        runner.load(str(checkpoint), map_location=device)
        policy = runner.get_inference_policy(device=device)
        obs = wrapped_env.get_observations()

        robot = raw_env.scene["robot"]
        ctrl_dt = float(raw_env.cfg.ctrl_dt)
        total_steps = round((warmup_s + duration_s) / ctrl_dt)
        warmup_steps = round(warmup_s / ctrl_dt)
        alive = np.ones(num_envs, dtype=np.bool_)
        speed_sum = np.zeros(num_envs, dtype=np.float64)
        lateral_sum = np.zeros(num_envs, dtype=np.float64)
        heading_error_sum = np.zeros(num_envs, dtype=np.float64)
        sample_count = np.zeros(num_envs, dtype=np.int64)
        heading_ref = np.asarray(robot.data.heading_w).copy()

        for step in range(total_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, infos = wrapped_env.step(actions)
            terminated = dones
            time_outs = infos.get("time_outs")
            if isinstance(time_outs, torch.Tensor):
                terminated = dones & ~time_outs
            alive &= ~terminated.detach().cpu().numpy()
            gravity = np.asarray(robot.data.projected_gravity_b)
            tilt = np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0))
            alive &= tilt <= math.radians(fall_tilt_deg)
            if step < warmup_steps:
                continue
            velocity = np.nan_to_num(np.asarray(robot.data.root_link_lin_vel_b))
            heading_error = np.abs(np_wrap_to_pi(np.asarray(robot.data.heading_w) - heading_ref))
            speed_sum[alive] += velocity[alive, 0]
            lateral_sum[alive] += np.abs(velocity[alive, 1])
            heading_error_sum[alive] += heading_error[alive]
            sample_count[alive] += 1

        sampled = sample_count > 0
        mean_speed = speed_sum[sampled] / sample_count[sampled]
        clean = alive & sampled
        return {
            "schema_version": 1,
            "task": "MicroduckSprintFlat",
            "checkpoint": str(checkpoint.resolve()),
            "backend": "mujoco",
            "device": device,
            "seed": seed,
            "num_envs": num_envs,
            "command_forward_m_s": speed,
            "warmup_s": warmup_s,
            "duration_s": duration_s,
            "fall_tilt_deg": fall_tilt_deg,
            "stress": stress,
            "survival_fraction": float(np.mean(alive)),
            "body_forward_speed_m_s": {
                "mean": float(np.mean(mean_speed)) if mean_speed.size else None,
                "p10": _quantile(mean_speed, 0.10),
                "median": _quantile(mean_speed, 0.50),
                "p90": _quantile(mean_speed, 0.90),
                "clean_survivor_mean": (
                    float(np.mean(speed_sum[clean] / sample_count[clean]))
                    if np.any(clean)
                    else None
                ),
            },
            "mean_absolute_lateral_speed_m_s": (
                float(np.mean(lateral_sum[sampled] / sample_count[sampled]))
                if np.any(sampled)
                else None
            ),
            "mean_absolute_heading_error_deg": (
                math.degrees(
                    float(np.mean(heading_error_sum[sampled] / sample_count[sampled]))
                )
                if np.any(sampled)
                else None
            ),
        }
    finally:
        if wrapped_env is not None:
            wrapped_env.close()
        else:
            raw_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--speed", type=float, default=2.2)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--fall-tilt-deg", type=float, default=70.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate(
        args.checkpoint,
        speed=args.speed,
        num_envs=args.num_envs,
        duration_s=args.duration,
        warmup_s=args.warmup,
        fall_tilt_deg=args.fall_tilt_deg,
        seed=args.seed,
        device=args.device,
        stress=args.stress,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(payload + "\n")


if __name__ == "__main__":
    main()
