"""Report per-leg joint kinematics for a Sprint checkpoint.

The speed battery in ``eval_sprint_speed.py`` scores the trunk and the head; it
cannot tell an alternating stride from a permanent split stance driven by the
shins alone. This script rolls a checkpoint out and reports each leg joint in a
*physical* convention where positive means "swung forward", then scores the
left-minus-right split of that angle. A human-like gait keeps the split
oscillating through zero -- thigh forward on one side is thigh back on the
other -- while a parked split stance holds it at a large constant.

MicroDuck's left and right leg frames are mirrored, so the raw joint signs are
opposite: forward-kinematics on ``scene_flat_bam.xml`` moves either ankle
forward for ``left_hip_pitch < 0`` and ``right_hip_pitch > 0``. ``LEG_SIGN``
below carries that measurement, so equal *physical* angles mean an equal
posture on both sides.

Usage:
    uv run --no-sync scripts/diagnose_gait.py model_2996.pt --speed 1.0
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
from unilab.utils.rotation import np_quat_apply_inverse, np_wrap_to_pi

from microduck_rl_unilab.conf_searchpath import register_conf_search_path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = package_root() / "conf" / "ppo"
LEG_JOINTS = ("hip_pitch", "knee", "ankle")
STEERING_JOINTS = (
    "left_hip_yaw",
    "right_hip_yaw",
    "left_hip_roll",
    "right_hip_roll",
)
HEAD_JOINTS = ("head_yaw", "head_roll")
FOOT_BODIES = ("ankle_left", "ankle_right")
# Sign that turns each side's raw joint angle into "swung forward is positive".
LEG_SIGN = {"left": -1.0, "right": 1.0}


def _compose_owner(
    *,
    speed: float,
    yaw_rate: float,
    duration_s: float,
    seed: int,
    owner: str,
    hold_spawn_heading: bool | None,
    heading_stiffness: float | None,
    heading_yaw_limit: float | None,
    cross_track_stiffness: float | None,
) -> Any:
    register_conf_search_path()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={owner}"])
    with open_dict(cfg):
        cfg.algo.seed = seed
        cfg.env.max_episode_seconds = duration_s + 1.0
        twist = cfg.env.commands.twist
        twist.ranges.lin_vel_x = [speed, speed]
        twist.ranges.lin_vel_y = [0.0, 0.0]
        twist.ranges.ang_vel_z = [yaw_rate, yaw_rate]
        if hold_spawn_heading is not None:
            twist.hold_spawn_heading = hold_spawn_heading
        if heading_stiffness is not None:
            twist.spawn_heading_stiffness = heading_stiffness
        if heading_yaw_limit is not None:
            twist.spawn_heading_yaw_rate_limit = heading_yaw_limit
        if cross_track_stiffness is not None:
            twist.spawn_cross_track_stiffness = cross_track_stiffness
        twist.rel_standing_envs = 0.0
        twist.rel_forward_envs = 0.0
        twist.turn_in_place_fraction = 0.0
        twist.resampling_time_range = [duration_s + 1.0, duration_s + 1.0]
        for name in cfg.env.curriculum:
            cfg.env.curriculum[name] = None
        for name in cfg.reward:
            cfg.reward[name] = None
    return cfg


def _mean_zero_crossings(series: np.ndarray, dt: float) -> float:
    """Mean crossings per second of each column's own mean, averaged over envs."""
    centred = series - np.mean(series, axis=0, keepdims=True)
    crossings = np.sum(np.diff(np.sign(centred), axis=0) != 0.0, axis=0)
    return float(np.mean(crossings) / (series.shape[0] * dt))


def _axis_report(series: np.ndarray, dt: float) -> dict[str, float]:
    """Summarize one joint column: offset, swing size, and crossing rate."""
    return {
        "mean_deg": math.degrees(float(np.mean(series))),
        "amplitude_deg": math.degrees(float(np.mean(np.std(series, axis=0)))),
        "min_deg": math.degrees(float(np.mean(np.min(series, axis=0)))),
        "max_deg": math.degrees(float(np.mean(np.max(series, axis=0)))),
        "crossings_per_s": _mean_zero_crossings(series, dt),
    }


def diagnose(
    checkpoint: Path,
    *,
    speed: float = 1.0,
    yaw_rate: float = 0.0,
    num_envs: int = 64,
    duration_s: float = 6.0,
    warmup_s: float = 1.0,
    seed: int = 123,
    device: str | None = None,
    owner: str = "microduck_sprint_flat/mujoco",
    hold_spawn_heading: bool | None = None,
    heading_stiffness: float | None = None,
    heading_yaw_limit: float | None = None,
    cross_track_stiffness: float | None = None,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if not 0.0 <= warmup_s < duration_s:
        raise ValueError("warmup_s must be in [0, duration_s)")

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = _compose_owner(
        speed=speed,
        yaw_rate=yaw_rate,
        duration_s=warmup_s + duration_s,
        seed=seed,
        owner=owner,
        hold_spawn_heading=hold_spawn_heading,
        heading_stiffness=heading_stiffness,
        heading_yaw_limit=heading_yaw_limit,
        cross_track_stiffness=cross_track_stiffness,
    )
    importlib.import_module("microduck_rl_unilab.tasks.microduck")
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=REPO_ROOT, algo_name="ppo").build_task_env_cfg_override()
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
        joint_names = [f"{side}_{joint}" for side in ("left", "right") for joint in LEG_JOINTS]
        joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
        steering_ids, _ = robot.find_joints(list(STEERING_JOINTS), preserve_order=True)
        head_ids, _ = robot.find_joints(list(HEAD_JOINTS), preserve_order=True)
        foot_ids, _ = robot.find_bodies(list(FOOT_BODIES), preserve_order=True)
        heading_ref = np.asarray(robot.data.heading_w).copy()

        joint_log: list[np.ndarray] = []
        steering_log: list[np.ndarray] = []
        head_log: list[np.ndarray] = []
        heading_log: list[np.ndarray] = []
        foot_xy_log: list[np.ndarray] = []
        body_velocity_log: list[np.ndarray] = []
        yaw_command_log: list[np.ndarray] = []
        height_log: list[np.ndarray] = []
        for step in range(total_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = wrapped_env.step(actions)
            if step < warmup_steps:
                continue
            joint_log.append(np.asarray(robot.data.joint_pos)[:, joint_ids].copy())
            steering_log.append(np.asarray(robot.data.joint_pos)[:, steering_ids].copy())
            head_log.append(np.asarray(robot.data.joint_pos)[:, head_ids].copy())
            heading_log.append(
                np_wrap_to_pi(np.asarray(robot.data.heading_w) - heading_ref).copy()
            )
            feet_w = np.asarray(robot.data.body_link_pos_w)[:, foot_ids, :]
            root = np.asarray(robot.data.root_link_pos_w)
            quat = np.asarray(robot.data.root_link_quat_w)
            left_b = np_quat_apply_inverse(quat, feet_w[:, 0, :] - root)
            right_b = np_quat_apply_inverse(quat, feet_w[:, 1, :] - root)
            foot_xy_log.append(np.stack([left_b[:, :2], right_b[:, :2]], axis=1))
            body_velocity_log.append(
                np.stack(
                    [
                        np.asarray(robot.data.root_link_lin_vel_b)[:, 1],
                        np.asarray(robot.data.root_link_ang_vel_b)[:, 2],
                    ],
                    axis=1,
                )
            )
            yaw_command_log.append(
                np.asarray(raw_env.command_manager.get_command("twist"))[:, 2].copy()
            )
            height_log.append(root[:, 2].copy())

        joints = np.nan_to_num(np.stack(joint_log, axis=0))
        steering = np.nan_to_num(np.stack(steering_log, axis=0))
        head = np.nan_to_num(np.stack(head_log, axis=0))
        heading = np.nan_to_num(np.stack(heading_log, axis=0))
        # Left columns come first, so flip them into the forward-positive frame.
        joints[:, :, : len(LEG_JOINTS)] *= LEG_SIGN["left"]
        joints[:, :, len(LEG_JOINTS) :] *= LEG_SIGN["right"]
        feet_xy = np.nan_to_num(np.stack(foot_xy_log, axis=0))
        body_velocity = np.nan_to_num(np.stack(body_velocity_log, axis=0))
        yaw_command = np.nan_to_num(np.stack(yaw_command_log, axis=0))
        heights = np.nan_to_num(np.stack(height_log, axis=0))

        report: dict[str, Any] = {
            "schema_version": 1,
            "checkpoint": str(checkpoint.resolve()),
            "command_forward_m_s": speed,
            "command_yaw_rate_rad_s": yaw_rate,
            "num_envs": num_envs,
            "sampled_seconds": duration_s,
            "ctrl_dt": ctrl_dt,
            "owner": owner,
            "trunk_height_m": {
                "mean": float(np.mean(heights)),
                "min": float(np.mean(np.min(heights, axis=0))),
            },
            "joints_forward_positive": {},
            "left_right_split": {},
        }
        for index, name in enumerate(joint_names):
            report["joints_forward_positive"][name] = _axis_report(joints[:, :, index], ctrl_dt)
        for index, joint in enumerate(LEG_JOINTS):
            split = joints[:, :, index] - joints[:, :, index + len(LEG_JOINTS)]
            report["left_right_split"][joint] = {
                # A parked split holds |mean| near the swing size; a real
                # alternating swing keeps |mean| small next to the amplitude
                # and spends about half the time on each sign.
                "mean_abs_deg": math.degrees(float(np.mean(np.abs(np.mean(split, axis=0))))),
                "amplitude_deg": math.degrees(float(np.mean(np.std(split, axis=0)))),
                "peak_to_peak_deg": math.degrees(float(np.mean(np.ptp(split, axis=0)))),
                "crossings_per_s": _mean_zero_crossings(split, ctrl_dt),
                "sign_flip_fraction": float(
                    np.mean(
                        np.minimum(
                            np.mean(split > 0.0, axis=0),
                            np.mean(split < 0.0, axis=0),
                        )
                    )
                ),
            }
            posture = 0.5 * (joints[:, :, index] + joints[:, :, index + len(LEG_JOINTS)])
            report["left_right_split"][joint]["shared_forward_offset_deg"] = math.degrees(
                float(np.mean(posture))
            )
        feet_x = feet_xy[:, :, :, 0]
        gap = feet_x[:, :, 0] - feet_x[:, :, 1]
        report["foot_forward_gap_m"] = {
            "mean_abs": float(np.mean(np.abs(np.mean(gap, axis=0)))),
            "amplitude": float(np.mean(np.std(gap, axis=0))),
            "peak_to_peak": float(np.mean(np.ptp(gap, axis=0))),
            "sign_flip_fraction": float(
                np.mean(
                    np.minimum(np.mean(gap > 0.0, axis=0), np.mean(gap < 0.0, axis=0))
                )
            ),
        }
        for side, index in (("left", 0), ("right", 1)):
            report[f"foot_{side}_forward_excursion_m"] = float(
                np.mean(np.ptp(feet_x[:, :, index], axis=0))
            )
        # Body-frame lateral geometry is independent of world heading. The
        # foot midpoint should stay near y=0 for a left/right-balanced policy.
        feet_y = feet_xy[:, :, :, 1]
        report["foot_lateral_m"] = {
            "left_mean": float(np.mean(feet_y[:, :, 0])),
            "right_mean": float(np.mean(feet_y[:, :, 1])),
            "midpoint_mean": float(np.mean(0.5 * (feet_y[:, :, 0] + feet_y[:, :, 1]))),
            "left_excursion": float(np.mean(np.ptp(feet_y[:, :, 0], axis=0))),
            "right_excursion": float(np.mean(np.ptp(feet_y[:, :, 1], axis=0))),
        }
        # These joints have mirrored frames. Report both raw angles and the
        # physical asymmetry combinations established by the FK probe:
        # equal physical hip-yaw/roll postures use opposite raw signs.
        report["steering_joints_deg"] = {
            name: math.degrees(float(np.mean(steering[:, :, index])))
            for index, name in enumerate(STEERING_JOINTS)
        }
        report["steering_asymmetry_deg"] = {
            "hip_yaw_physical_difference": math.degrees(
                float(np.mean(steering[:, :, 0] + steering[:, :, 1]))
            ),
            "hip_roll_physical_difference": math.degrees(
                float(np.mean(steering[:, :, 2] + steering[:, :, 3]))
            ),
        }

        # Signed per-second means. The averages above hide a one-sided drift,
        # and a veer that starts partway into the run looks the same as a
        # steady one. Positive head_yaw / heading is a turn to the robot's
        # left, so a leftward curve shows as both columns growing positive.
        steps_per_second = max(1, round(1.0 / ctrl_dt))
        seconds = head.shape[0] // steps_per_second
        timeline: list[dict[str, float]] = []
        for index in range(seconds):
            window = slice(index * steps_per_second, (index + 1) * steps_per_second)
            timeline.append(
                {
                    "second": float(warmup_s + index),
                    "head_yaw_signed_deg": math.degrees(float(np.mean(head[window, :, 0]))),
                    "head_roll_signed_deg": math.degrees(float(np.mean(head[window, :, 1]))),
                    "heading_signed_deg": math.degrees(float(np.mean(heading[window]))),
                    "heading_abs_deg": math.degrees(float(np.mean(np.abs(heading[window])))),
                    "lateral_velocity_signed_m_s": float(
                        np.mean(body_velocity[window, :, 0])
                    ),
                    "yaw_rate_signed_deg_s": math.degrees(
                        float(np.mean(body_velocity[window, :, 1]))
                    ),
                }
            )
        report["timeline_per_second"] = timeline
        report["head_yaw_signed_deg"] = math.degrees(float(np.mean(head[:, :, 0])))
        report["head_roll_signed_deg"] = math.degrees(float(np.mean(head[:, :, 1])))
        report["heading_signed_deg"] = math.degrees(float(np.mean(heading)))
        report["lateral_velocity_signed_m_s"] = float(np.mean(body_velocity[:, :, 0]))
        report["yaw_rate_signed_deg_s"] = math.degrees(
            float(np.mean(body_velocity[:, :, 1]))
        )
        yaw_actual = body_velocity[:, :, 1]
        yaw_residual = yaw_actual - yaw_command
        report["yaw_tracking"] = {
            "command_abs_mean_rad_s": float(np.mean(np.abs(yaw_command))),
            "actual_abs_mean_rad_s": float(np.mean(np.abs(yaw_actual))),
            "residual_abs_mean_rad_s": float(np.mean(np.abs(yaw_residual))),
        }
        active = np.abs(yaw_command) > 0.05
        if np.any(active):
            report["yaw_tracking"]["same_sign_fraction"] = float(
                np.mean(yaw_command[active] * yaw_actual[active] > 0.0)
            )
            report["yaw_tracking"]["command_opposes_heading_fraction"] = float(
                np.mean(yaw_command[active] * heading[active] < 0.0)
            )
            command_active = yaw_command[active]
            report["yaw_tracking"]["response_gain"] = float(
                np.dot(command_active, yaw_actual[active])
                / max(float(np.dot(command_active, command_active)), 1.0e-12)
            )
        # Share of environments whose final heading sits on the same side, so a
        # systematic veer is separated from per-episode noise.
        final = heading[-steps_per_second:].mean(axis=0)
        report["heading_left_fraction"] = float(np.mean(final > 0.0))
        return report
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
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--owner",
        default="microduck_sprint_flat/mujoco",
        help="Hydra task owner used to build the diagnostic environment.",
    )
    parser.add_argument(
        "--hold-spawn-heading",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the owner's spawn-heading feedback setting.",
    )
    parser.add_argument("--heading-stiffness", type=float, default=None)
    parser.add_argument("--heading-yaw-limit", type=float, default=None)
    parser.add_argument("--cross-track-stiffness", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = diagnose(
        args.checkpoint,
        speed=args.speed,
        yaw_rate=args.yaw_rate,
        num_envs=args.num_envs,
        duration_s=args.duration,
        warmup_s=args.warmup,
        seed=args.seed,
        device=args.device,
        owner=args.owner,
        hold_spawn_heading=args.hold_spawn_heading,
        heading_stiffness=args.heading_stiffness,
        heading_yaw_limit=args.heading_yaw_limit,
        cross_track_stiffness=args.cross_track_stiffness,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(payload + "\n")


if __name__ == "__main__":
    main()
