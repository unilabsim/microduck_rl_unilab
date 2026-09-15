"""Pollen Robotics MicroDuck velocity task on the Manager-Based runtime."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, make_manager_based_rl_env

from microduck_rl_unilab.assets import ensure_microduck_assets


def make_microduck_velocity_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ManagerBasedRlEnv:
    """Resolve MicroDuck STL assets before materializing the generic runtime."""
    ensure_microduck_assets()
    return make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)


registry.register_env_config("MicroduckVelocityFlat", ManagerBasedRlEnvCfg)
registry.register_env(
    "MicroduckVelocityFlat",
    make_microduck_velocity_env,
    sim_backend="mujoco",
)
registry.register_env(
    "MicroduckVelocityFlat",
    make_microduck_velocity_env,
    sim_backend="mjwarp",
)

# Forward-only max-speed policy. The Hydra owner specializes the velocity
# contract; the generic runtime and cold-path asset factory remain shared.
registry.register_env_config("MicroduckSprintFlat", ManagerBasedRlEnvCfg)
registry.register_env(
    "MicroduckSprintFlat",
    make_microduck_velocity_env,
    sim_backend="mujoco",
)

# VelStand (walking + fall recovery) on the ground-contact BAM model.
registry.register_env_config("MicroduckVelstandFlat", ManagerBasedRlEnvCfg)
registry.register_env(
    "MicroduckVelstandFlat",
    make_microduck_velocity_env,
    sim_backend="mujoco",
)

# Standup (sit/ground-pose mixed resets -> stand + body-pose tracking) on the
# ground-contact BAM model.
registry.register_env_config("MicroduckStandupFlat", ManagerBasedRlEnvCfg)
registry.register_env(
    "MicroduckStandupFlat",
    make_microduck_velocity_env,
    sim_backend="mujoco",
)

# The minimal command/term owners deliberately share the same generic
# ManagerBasedRlEnv factory.  Their behavior is selected entirely by the
# Hydra owner (command, metrics, recorder, and reward terms); the runtime does
# not branch on task names.
for _task_name in ("MicroduckGroundPickFlat", "MicroduckSitStandFlat"):
    registry.register_env_config(_task_name, ManagerBasedRlEnvCfg)
    registry.register_env(_task_name, make_microduck_velocity_env, sim_backend="mjwarp")


__all__ = ["make_microduck_velocity_env"]
