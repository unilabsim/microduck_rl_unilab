"""Contracts for the forward-only MicroduckSprintFlat recipe."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.cli import package_root
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.managers import CurriculumTermCfg, RewardTermCfg, SceneEntityCfg
from unilab.managers._types import ManagerBasedRlEnv

from microduck_rl_unilab.tasks.microduck.sprint_terms import (
    heading_hold,
    running_command_ranges_curriculum,
    running_forward_progress,
    running_planar_drift_cost,
)
from microduck_rl_unilab.tasks.microduck.manager_terms import MicroduckVelocityCommandCfg
from microduck_rl_unilab.tasks.microduck.deploy_contract import (
    MICRODUCK_ACTOR_OBS_DIM,
    MICRODUCK_CRITIC_OBS_DIM,
    MICRODUCK_NUM_ACTION,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
CONF_DIR = package_root() / "conf" / "ppo"


def _materialize_owner() -> tuple[Any, ManagerBasedRlEnvCfg]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=["task=microduck_sprint_flat/mujoco"])
    registry.ensure_registries()
    env_cfg = registry.materialize_env_config("MicroduckSprintFlat")
    apply_cfg_overrides(
        env_cfg,
        BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo").build_task_env_cfg_override(),
    )
    env_cfg.validate()
    return cfg, env_cfg


def test_sprint_owner_specializes_velocity_contract() -> None:
    cfg, env_cfg = _materialize_owner()
    assert cfg.training.task_name == "MicroduckSprintFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.max_iterations == 11750
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(0.02)

    assert env_cfg.max_episode_seconds == pytest.approx(12.0)
    twist = env_cfg.commands["twist"]
    assert twist.resampling_time_range == [12.0, 12.0]
    assert twist.ranges.lin_vel_x == [0.2, 0.45]
    assert twist.ranges.lin_vel_y == [-0.02, 0.02]
    assert twist.ranges.ang_vel_z == [-0.05, 0.05]
    assert twist.rel_standing_envs == pytest.approx(0.03)
    assert twist.turn_in_place_fraction == pytest.approx(0.0)

    assert env_cfg.events["push_robot"] is None
    for name in (
        "head_pose_bias_weight",
        "standing_envs",
        "head_pose_range",
        "base_com_range",
        "head_com_range",
    ):
        assert env_cfg.curriculum[name] is None
    stages = env_cfg.curriculum["running_speed_range"].params["speed_stages"]
    assert stages[0] == {"step": 0, "min_speed": 0.2, "max_speed": 0.45}
    assert stages[-1] == {"step": 264000, "min_speed": 1.65, "max_speed": 2.2}

    rewards = env_cfg.rewards
    assert rewards["forward_progress"].func is running_forward_progress
    assert rewards["forward_progress"].weight == pytest.approx(5.0)
    assert rewards["forward_progress"].params["speed_cap"] == pytest.approx(2.4)
    assert rewards["planar_drift"].func is running_planar_drift_cost
    assert rewards["heading_hold"].func is heading_hold
    assert rewards["heading_hold"].weight == pytest.approx(1.5)
    assert rewards["tracking_ang_vel"].weight == pytest.approx(0.5)
    assert rewards["leg_pose"].weight == pytest.approx(0.15)
    assert rewards["upright"].weight == pytest.approx(0.75)
    assert rewards["air_time"].weight == pytest.approx(0.0)
    assert rewards["head_pose_tracking"].weight == pytest.approx(0.0)
    assert rewards["action_rate"].weight == pytest.approx(-0.02)


def test_sprint_robust_owner_pins_speed_and_restores_disturbances() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=["task=microduck_sprint_robust_flat/mujoco"])
    registry.ensure_registries()
    env_cfg = registry.materialize_env_config("MicroduckSprintFlat")
    apply_cfg_overrides(
        env_cfg,
        BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo").build_task_env_cfg_override(),
    )
    env_cfg.validate()

    assert cfg.training.task_name == "MicroduckSprintFlat"
    assert cfg.algo.max_iterations == 100
    twist = env_cfg.commands["twist"]
    assert twist.ranges.lin_vel_x == [1.65, 2.2]
    assert env_cfg.curriculum["running_speed_range"] is None
    assert env_cfg.curriculum["action_rate_weight"] is None
    assert env_cfg.rewards["action_rate"].weight == pytest.approx(-0.10)
    push = env_cfg.events["push_robot"]
    assert push is not None
    assert push.mode == "interval"
    assert push.params["velocity_range"]["x"] == [-0.03, 0.03]
    pose = env_cfg.events["reset_base"].params["pose_range"]
    assert pose["roll"][1] == pytest.approx(math.radians(1.0))
    assert pose["pitch"][1] == pytest.approx(math.radians(1.0))


def test_sprint_registry_is_mujoco_only() -> None:
    registry.ensure_registries()
    assert registry.list_registered_envs()["MicroduckSprintFlat"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco"],
    }


class _FakeEntity:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            root_link_lin_vel_b=np.asarray(
                [[-1.0, 0.1, 0.0], [0.7, -0.2, 0.0], [3.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            root_link_ang_vel_b=np.asarray(
                [[0.0, 0.0, 0.2], [0.0, 0.0, -0.1], [0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            heading_w=np.asarray([0.0, 0.2, np.pi - 0.1], dtype=np.float32),
        )


def _fake_env() -> ManagerBasedRlEnv:
    entity = _FakeEntity()
    command = np.asarray(
        [[0.4, 0.0, 0.0], [0.8, -0.1, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=3,
            scene={"robot": entity},
            command_manager=SimpleNamespace(get_command=lambda name: command),
            episode_length_buf=np.asarray([2, 2, 2], dtype=np.int64),
        ),
    )


def test_sprint_rewards_match_reference_math_and_wrap_heading() -> None:
    env = _fake_env()
    np.testing.assert_allclose(running_forward_progress(env, speed_cap=1.4), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        running_planar_drift_cost(env, lateral_weight=4.0),
        [0.08, 0.05, 0.0],
        atol=1e-6,
    )

    term = heading_hold(
        RewardTermCfg(
            func=heading_hold,
            weight=1.0,
            params={"std": 0.4, "asset_cfg": SceneEntityCfg("robot")},
        ),
        env,
    )
    cast(Any, env.scene["robot"]).data.heading_w[:] = [0.4, -0.2, -np.pi + 0.1]
    expected = np.exp(-np.square([0.4, -0.4, 0.2]) / 0.4**2)
    np.testing.assert_allclose(term(env), expected, rtol=1e-5)


def test_running_speed_curriculum_mutates_only_forward_band() -> None:
    term_cfg = MicroduckVelocityCommandCfg(
        entity_name="robot",
        ranges=MicroduckVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.45),
            lin_vel_y=(-0.02, 0.02),
            ang_vel_z=(-0.05, 0.05),
        ),
        resampling_time_range=(12.0, 12.0),
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            common_step_counter=24,
            command_manager=SimpleNamespace(get_term_cfg=lambda name: term_cfg),
        ),
    )
    stages = [
        {"step": 0, "min_speed": 0.2, "max_speed": 0.45},
        {"step": 24, "min_speed": 0.3, "max_speed": 0.55},
    ]
    cfg = CurriculumTermCfg(
        func=running_command_ranges_curriculum,
        params={"command_name": "twist", "speed_stages": stages},
    )
    term = running_command_ranges_curriculum(cfg, env)
    assert term(env, slice(None), "twist", stages) == {
        "min_speed": 0.3,
        "max_speed": 0.55,
    }
    assert term_cfg.ranges.lin_vel_x == (0.3, 0.55)
    assert term_cfg.ranges.lin_vel_y == (-0.02, 0.02)
    assert term_cfg.ranges.ang_vel_z == (-0.05, 0.05)


@pytest.mark.slow
def test_sprint_owner_builds_and_steps_real_mujoco_env() -> None:
    pytest.importorskip("mujoco")
    cfg, _ = _materialize_owner()
    env = cast(
        Any,
        registry.make(
            "MicroduckSprintFlat",
            sim_backend="mujoco",
            num_envs=1,
            env_cfg_override=BackendAdapter(
                cfg,
                root_dir=ROOT_DIR,
                algo_name="ppo",
            ).build_task_env_cfg_override(),
        ),
    )
    try:
        obs, info = env.reset()
        assert isinstance(info, dict)
        assert obs["obs"].shape == (1, MICRODUCK_ACTOR_OBS_DIM)
        assert obs["critic"].shape == (1, MICRODUCK_CRITIC_OBS_DIM)
        state = env.step(np.zeros((1, MICRODUCK_NUM_ACTION), dtype=np.float32))
        assert np.isfinite(state.reward).all()
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert env.curriculum_manager.active_terms == [
            "action_rate_weight",
            "running_speed_range",
        ]
        assert "push_robot" not in env.event_manager.active_terms
    finally:
        env.close()
