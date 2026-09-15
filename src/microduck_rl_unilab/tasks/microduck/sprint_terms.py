"""Sprint-specific reward terms for the MicroDuck Manager-Based task."""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.mdp import UniformVelocityCommandCfg
from unilab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse, np_wrap_to_pi
from microduck_rl_unilab.tasks.microduck.manager_terms import (
    _FootContactTerm,
    _asset_selection,
    _command,
)

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=("jaw_soft",))
# MicroDuck's left and right leg frames are mirrored, so a shared joint sign
# means a mirrored posture rather than a matching one. Forward kinematics on
# the cold path (``scripts/diagnose_gait.py`` docstring records the probe)
# moves either ankle forward for ``left_hip_pitch < 0`` and
# ``right_hip_pitch > 0``. Terms that compare the two legs must apply these
# signs first, otherwise "left minus right" measures the crouch depth instead
# of the front-to-back split between the thighs.
_LEFT_RIGHT_FORWARD_SIGN = (-1.0, 1.0)
# Measured on the cold path at the home keyframe (neck/head pitch at +20 deg):
# the head body's local +x axis coincides with world up, so it is the "top of
# the head" direction for MicroDuck.
_DEFAULT_HEAD_UP_AXIS_B = (1.0, 0.0, 0.0)


def _positive_real(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def running_forward_progress(
    env: ManagerBasedRlEnv,
    speed_cap: float = 1.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Reward capped positive body-forward speed with a linear gradient."""
    cap = _positive_real(speed_cap, label="running_forward_progress speed_cap")
    asset = cast("Entity", env.scene[asset_cfg.name])
    velocity_x = np.nan_to_num(
        asset.data.root_link_lin_vel_b[:, 0],
        nan=0.0,
        posinf=cap,
        neginf=0.0,
    )
    return np.asarray(np.clip(velocity_x, 0.0, cap) / cap, dtype=get_global_dtype())


def running_planar_drift_cost(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    lateral_weight: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Penalize body-frame lateral drift and yaw-rate command error."""
    weight = _positive_real(lateral_weight, label="running_planar_drift_cost lateral_weight")
    if not isinstance(command_name, str) or not command_name:
        raise ValueError("running_planar_drift_cost command_name must be non-empty")
    command = np.asarray(env.command_manager.get_command(command_name))
    if command.shape != (env.num_envs, 3):
        raise ValueError(
            f"running_planar_drift_cost command must have shape ({env.num_envs}, 3), "
            f"received {command.shape}"
        )
    asset = cast("Entity", env.scene[asset_cfg.name])
    lateral_error = np.nan_to_num(
        asset.data.root_link_lin_vel_b[:, 1] - command[:, 1],
        nan=0.0,
    )
    yaw_error = np.nan_to_num(
        asset.data.root_link_ang_vel_b[:, 2] - command[:, 2],
        nan=0.0,
    )
    return np.asarray(
        np.square(yaw_error) + weight * np.square(lateral_error),
        dtype=get_global_dtype(),
    )


def _bind_heading_asset(cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv, *, label: str):
    asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    if not isinstance(asset_cfg, SceneEntityCfg):
        raise TypeError(f"{label} asset_cfg must be SceneEntityCfg")
    entity = cast("Entity", env.scene[asset_cfg.name])
    heading_ref = np.asarray(entity.data.heading_w, dtype=get_global_dtype()).copy()
    return entity, heading_ref


def _refresh_heading_ref(
    env: ManagerBasedRlEnv,
    entity: "Entity",
    heading_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    heading = np.asarray(entity.data.heading_w, dtype=get_global_dtype())
    fresh = env.episode_length_buf <= 1
    heading_ref[fresh] = heading[fresh]
    return heading, np_wrap_to_pi(heading - heading_ref)


class heading_hold(ManagerTermBase):
    """Reward holding each environment's spawn heading."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        allowed = {"std", "asset_cfg"}
        unexpected = set(cfg.params) - allowed
        if unexpected:
            raise TypeError(f"heading_hold received unsupported parameters: {sorted(unexpected)}")
        self._std = _positive_real(cfg.params.get("std", 0.4), label="heading_hold std")
        self._entity, self._heading_ref = _bind_heading_asset(cfg, env, label="heading_hold")

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._heading_ref[ids] = self._entity.data.heading_w[ids]

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        _, error = _refresh_heading_ref(env, self._entity, self._heading_ref)
        return np.asarray(np.exp(-np.square(error) / self._std**2), dtype=get_global_dtype())


class heading_abs_cost(ManagerTermBase):
    """Linear spawn-heading error. Keeps a gradient after the Gaussian bowl dies."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        unexpected = set(cfg.params) - {"asset_cfg"}
        if unexpected:
            raise TypeError(f"heading_abs_cost received unsupported parameters: {sorted(unexpected)}")
        self._entity, self._heading_ref = _bind_heading_asset(cfg, env, label="heading_abs_cost")

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._heading_ref[ids] = self._entity.data.heading_w[ids]

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        _, error = _refresh_heading_ref(env, self._entity, self._heading_ref)
        return np.asarray(np.abs(error), dtype=get_global_dtype())


def running_straightness_cost(
    env: ManagerBasedRlEnv,
    lateral_scale: float = 0.20,
    yaw_rate_scale: float = 0.20,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Linear cost for residual lateral velocity and yaw rate.

    ``running_planar_drift_cost`` is quadratic. A persistent 0.04 rad/s yaw
    rate barely registers per step, yet accumulates roughly 25 degrees over a
    ten-second rollout. This term deliberately retains a constant gradient
    near zero so a slow curve is not a cheap substitute for a straight run.

    When ``command_name`` is set, the cost is against the twist command so a
    spawn-heading yaw correction is not punished for turning the right way.
    """
    lateral_cap = _positive_real(
        lateral_scale, label="running_straightness_cost lateral_scale"
    )
    yaw_cap = _positive_real(
        yaw_rate_scale, label="running_straightness_cost yaw_rate_scale"
    )
    asset = cast("Entity", env.scene[asset_cfg.name])
    lateral = np.nan_to_num(asset.data.root_link_lin_vel_b[:, 1], nan=0.0)
    yaw_rate = np.nan_to_num(asset.data.root_link_ang_vel_b[:, 2], nan=0.0)
    if command_name:
        if not isinstance(command_name, str):
            raise TypeError("running_straightness_cost command_name must be str")
        command = np.asarray(env.command_manager.get_command(command_name))
        if command.shape != (env.num_envs, 3):
            raise ValueError(
                "running_straightness_cost command must have shape "
                f"({env.num_envs}, 3), received {command.shape}"
            )
        lateral = lateral - command[:, 1]
        yaw_rate = yaw_rate - command[:, 2]
    cost = 0.5 * (
        np.clip(np.abs(lateral) / lateral_cap, 0.0, 1.0)
        + np.clip(np.abs(yaw_rate) / yaw_cap, 0.0, 1.0)
    )
    return np.asarray(cost, dtype=get_global_dtype())


class cross_track_abs_cost(ManagerTermBase):
    """Linear lateral displacement from each episode's spawn-heading ray.

    Instantaneous heading and yaw-rate rewards do not remember that many small
    errors form a large arc. Cross-track displacement is that missing memory:
    later steps become increasingly expensive while the policy curves away,
    which makes a longer episode an actual straight-line curriculum rather
    than merely a longer constant command.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        allowed = {"distance_cap", "asset_cfg"}
        unexpected = set(cfg.params) - allowed
        if unexpected:
            raise TypeError(
                f"cross_track_abs_cost received unsupported parameters: {sorted(unexpected)}"
            )
        self._cap = _positive_real(
            cfg.params.get("distance_cap", 0.30),
            label="cross_track_abs_cost distance_cap",
        )
        self._entity, self._heading_ref = _bind_heading_asset(
            cfg, env, label="cross_track_abs_cost"
        )
        self._origin_xy = np.asarray(
            self._entity.data.root_link_pos_w[:, :2],
            dtype=get_global_dtype(),
        ).copy()

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._heading_ref[ids] = self._entity.data.heading_w[ids]
        self._origin_xy[ids] = self._entity.data.root_link_pos_w[ids, :2]

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        _refresh_heading_ref(env, self._entity, self._heading_ref)
        fresh = env.episode_length_buf <= 1
        position = np.asarray(
            self._entity.data.root_link_pos_w[:, :2],
            dtype=get_global_dtype(),
        )
        self._origin_xy[fresh] = position[fresh]
        delta = np.nan_to_num(position - self._origin_xy, nan=0.0)
        lateral = -delta[:, 0] * np.sin(self._heading_ref) + delta[
            :, 1
        ] * np.cos(self._heading_ref)
        return np.asarray(
            np.clip(np.abs(lateral) / self._cap, 0.0, 1.0),
            dtype=get_global_dtype(),
        )


class head_facing_hold(ManagerTermBase):
    """Keep the head in the sagittal plane; pitch nod is free.

    ``neck_pitch`` / ``head_pitch`` may look up or down so the bird can reach
    forward and balance. ``head_yaw`` and ``head_roll`` stay centered so the
    face does not turn or tilt left/right. Pitch-related params are accepted
    for Hydra merge compatibility and ignored.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        allowed = {
            "asset_cfg",
            "command_name",
            "yaw_std",
            "roll_std",
            "pitch_tuck_std",
            "extra_slack",
            "extra_std",
            "forward_pitch_sign",
        }
        unexpected = {
            key for key in (set(cfg.params) - allowed) if cfg.params.get(key) is not None
        }
        if unexpected:
            raise TypeError(f"head_facing_hold received unsupported parameters: {sorted(unexpected)}")
        self._entity, self._joint_ids = _asset_selection(cfg, env, term="head_facing_hold")
        if self._joint_ids.size != 4:
            raise ValueError(
                "head_facing_hold expects [neck_pitch, head_pitch, head_yaw, head_roll]"
            )
        command_name = cfg.params.get("command_name")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError("head_facing_hold command_name must be a non-empty string")
        self._command_name = command_name
        self._yaw_std = _positive_real(cfg.params.get("yaw_std", 0.15), label="head_facing_hold yaw_std")
        self._roll_std = _positive_real(
            cfg.params.get("roll_std", 0.12), label="head_facing_hold roll_std"
        )
        self._pitch_tuck_std = _positive_real(
            cfg.params.get("pitch_tuck_std", 0.25),
            label="head_facing_hold pitch_tuck_std",
        )
        self._extra_slack = float(cfg.params.get("extra_slack", 0.45))
        if not math.isfinite(self._extra_slack) or self._extra_slack < 0.0:
            raise ValueError("head_facing_hold extra_slack must be finite and >= 0")
        self._extra_std = _positive_real(
            cfg.params.get("extra_std", 0.35), label="head_facing_hold extra_std"
        )
        sign = cfg.params.get("forward_pitch_sign", 1.0)
        if isinstance(sign, bool) or not isinstance(sign, Real):
            raise TypeError("head_facing_hold forward_pitch_sign must be a real number")
        self._forward_sign = float(sign)
        if self._forward_sign not in (-1.0, 1.0):
            raise ValueError("head_facing_hold forward_pitch_sign must be +1 or -1")

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        command = _command(env, "head_facing_hold", self._command_name)
        if command.shape[1] != 4:
            raise ValueError(
                f"head_facing_hold command must have width 4, received {command.shape}"
            )
        actual = self._entity.data.joint_pos[:, self._joint_ids]
        default = self._entity.data.default_joint_pos[:, self._joint_ids]
        error = actual - default - command
        terms = np.concatenate(
            [
                np.exp(-np.square(error[:, 2:3] / self._yaw_std)),
                np.exp(-np.square(error[:, 3:4] / self._roll_std)),
            ],
            axis=1,
        )
        return np.asarray(np.mean(terms, axis=1), dtype=get_global_dtype())


class head_facing_abs_cost(ManagerTermBase):
    """Linear ``head_yaw`` / ``head_roll`` error. Keeps a gradient at the joint stops.

    ``head_facing_hold`` is a Gaussian bowl, so once the policy parks a head
    joint against its mechanical limit (``head_roll`` stops at 25 deg) the bowl
    is numerically zero and stops teaching. This term adds a constant slope that
    still pulls the target back toward the sagittal plane from saturation.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        allowed = {"asset_cfg", "command_name"}
        unexpected = {
            key for key in (set(cfg.params) - allowed) if cfg.params.get(key) is not None
        }
        if unexpected:
            raise TypeError(
                f"head_facing_abs_cost received unsupported parameters: {sorted(unexpected)}"
            )
        self._entity, self._joint_ids = _asset_selection(cfg, env, term="head_facing_abs_cost")
        if self._joint_ids.size != 2:
            raise ValueError("head_facing_abs_cost expects [head_yaw, head_roll]")

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del env, params
        error = (
            self._entity.data.joint_pos[:, self._joint_ids]
            - self._entity.data.default_joint_pos[:, self._joint_ids]
        )
        cost = np.mean(np.abs(np.nan_to_num(error, nan=0.0)), axis=1)
        return np.asarray(cost, dtype=get_global_dtype())


class joint_band_cost(ManagerTermBase):
    """Linear cost for leaving a per-joint comfort band, in degrees.

    ``head_upright_hold`` only constrains where the head points, which the
    sprint policy satisfied by folding ``neck_pitch`` onto its -90 deg stop and
    counter-twisting ``head_pitch`` back up: the head-top faced the sky with
    both joints jammed at the end of travel. This term keeps the pitch chain
    inside a usable band so the forward reach comes from a moderate nod. The
    cost is linear in the overshoot, so it keeps teaching at the stops.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        allowed = {"asset_cfg", "band_deg"}
        unexpected = {
            key for key in (set(cfg.params) - allowed) if cfg.params.get(key) is not None
        }
        if unexpected:
            raise TypeError(
                f"joint_band_cost received unsupported parameters: {sorted(unexpected)}"
            )
        self._entity, self._joint_ids = _asset_selection(cfg, env, term="joint_band_cost")
        bands = cfg.params.get("band_deg")
        if bands is None:
            raise ValueError("joint_band_cost band_deg is required")
        band_array = np.asarray(bands, dtype=np.float64)
        if band_array.shape != (self._joint_ids.size, 2) or not np.isfinite(band_array).all():
            raise ValueError(
                "joint_band_cost band_deg must be one finite [lower, upper] pair per "
                f"selected joint, received shape {band_array.shape}"
            )
        if not np.all(band_array[:, 0] < band_array[:, 1]):
            raise ValueError("joint_band_cost band_deg requires lower < upper")
        radians = np.radians(band_array)
        self._lower = np.asarray(radians[None, :, 0], dtype=get_global_dtype())
        self._upper = np.asarray(radians[None, :, 1], dtype=get_global_dtype())

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del env, params
        actual = np.nan_to_num(self._entity.data.joint_pos[:, self._joint_ids], nan=0.0)
        overshoot = np.maximum(actual - self._upper, 0.0) + np.maximum(self._lower - actual, 0.0)
        return np.asarray(np.mean(overshoot, axis=1), dtype=get_global_dtype())


def _moving_gate(
    env: ManagerBasedRlEnv,
    *,
    term: str,
    command_name: str,
    command_threshold: float,
) -> np.ndarray:
    command = _command(env, term, command_name)
    moving = (
        np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2]) > command_threshold
    )
    return moving.astype(get_global_dtype())


class running_alternate_steps(_FootContactTerm):
    """Human running contact: one stance foot, or a brief flight. Never a shuffle.

    Single stance scores 1. Flight (both feet off) scores ``flight_value`` so a
    real run is legal. Double stance scores 0, which is the limp / shuffle that
    produced the leftward drift.
    """

    _allowed_params = frozenset({"command_name", "command_threshold", "flight_value"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command_name = cfg.params.get("command_name", "twist")
        if not isinstance(self._command_name, str) or not self._command_name:
            raise ValueError("running_alternate_steps command_name must be a non-empty string")
        self._command_threshold = _positive_real(
            cfg.params.get("command_threshold", 0.01),
            label="running_alternate_steps command_threshold",
        )
        flight = cfg.params.get("flight_value", 0.6)
        if isinstance(flight, bool) or not isinstance(flight, Real):
            raise TypeError("running_alternate_steps flight_value must be a real number")
        self._flight = float(flight)
        if not 0.0 <= self._flight <= 1.0:
            raise ValueError("running_alternate_steps flight_value must be in [0, 1]")

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        n_contact = np.sum(self._contact(env).astype(np.int32), axis=1)
        reward = np.where(
            n_contact == 1,
            1.0,
            np.where(n_contact == 0, self._flight, 0.0),
        )
        gate = _moving_gate(
            env,
            term="running_alternate_steps",
            command_name=self._command_name,
            command_threshold=self._command_threshold,
        )
        return np.asarray(reward * gate, dtype=get_global_dtype())


class running_contact_symmetry_cost(_FootContactTerm):
    """Penalize left/right stance-duty mismatch, the limp that steers the run.

    An exponential moving average of each foot's contact bit stays near 0.5
    on a symmetric alternate gait and splits toward 0/1 on a one-legged limp.
    """

    _allowed_params = frozenset({"command_name", "command_threshold", "tau_s"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command_name = cfg.params.get("command_name", "twist")
        if not isinstance(self._command_name, str) or not self._command_name:
            raise ValueError(
                "running_contact_symmetry_cost command_name must be a non-empty string"
            )
        self._command_threshold = _positive_real(
            cfg.params.get("command_threshold", 0.01),
            label="running_contact_symmetry_cost command_threshold",
        )
        tau = _positive_real(
            cfg.params.get("tau_s", 0.4),
            label="running_contact_symmetry_cost tau_s",
        )
        self._alpha = min(1.0, float(env.step_dt) / tau)
        self._duty = np.full((env.num_envs, 2), 0.5, dtype=get_global_dtype())

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._duty[ids] = 0.5

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        contact = self._contact(env).astype(get_global_dtype())
        self._duty = (1.0 - self._alpha) * self._duty + self._alpha * contact
        cost = np.abs(self._duty[:, 0] - self._duty[:, 1])
        gate = _moving_gate(
            env,
            term="running_contact_symmetry_cost",
            command_name=self._command_name,
            command_threshold=self._command_threshold,
        )
        return np.asarray(cost * gate, dtype=get_global_dtype())


class _SwingOscillationTerm(ManagerTermBase):
    """Score how far a left-minus-right signal swings around its own average.

    Both gait terms below share one problem. The quantity that separates a run
    from a parked split stance is not how large the left/right difference is --
    a frozen split holds a very large one -- but whether that difference keeps
    swapping sign. So each term tracks an exponential moving average over
    roughly one stride and scores ``|signal| - |average|``: a symmetric swing
    averages to zero and keeps its full magnitude, while a constant offset
    cancels itself and scores nothing.
    """

    def __init__(
        self,
        cfg: ManagerTermBaseCfg,
        env: ManagerBasedRlEnv,
        *,
        label: str,
        default_cap: float,
        cap_key: str,
    ):
        super().__init__(env)
        self._label = label
        self._command_name = cfg.params.get("command_name", "twist")
        if not isinstance(self._command_name, str) or not self._command_name:
            raise ValueError(f"{label} command_name must be a non-empty string")
        self._command_threshold = _positive_real(
            cfg.params.get("command_threshold", 0.01),
            label=f"{label} command_threshold",
        )
        self._cap = _positive_real(cfg.params.get(cap_key, default_cap), label=f"{label} {cap_key}")
        tau = _positive_real(cfg.params.get("tau_s", 0.5), label=f"{label} tau_s")
        self._alpha = min(1.0, float(env.step_dt) / tau)
        self._average = np.zeros(env.num_envs, dtype=get_global_dtype())

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._average[ids] = 0.0

    def _score(self, env: ManagerBasedRlEnv, signal: np.ndarray) -> np.ndarray:
        signal = np.nan_to_num(signal, nan=0.0)
        self._average = (1.0 - self._alpha) * self._average + self._alpha * signal
        swing = np.abs(signal) - np.abs(self._average)
        reward = np.clip(swing / self._cap, 0.0, 1.0)
        gate = _moving_gate(
            env,
            term=self._label,
            command_name=self._command_name,
            command_threshold=self._command_threshold,
        )
        return np.asarray(reward * gate, dtype=get_global_dtype())


class running_leg_alternation(_SwingOscillationTerm):
    """Reward the thighs trading places front to back.

    ``hip_pitch`` is read in the mirrored-corrected frame of
    ``_LEFT_RIGHT_FORWARD_SIGN``, so the signal is the left thigh's forward
    angle minus the right thigh's. Human running drives that split through zero
    twice per stride. The reported failure -- left thigh parked forward, right
    thigh parked back, all motion in the shins and ankles -- holds the split at
    a large constant, which this term scores as zero. Contact-based terms
    cannot see the difference, because a parked split still takes turns
    touching the ground.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        allowed = {"asset_cfg", "command_name", "command_threshold", "swing_cap_deg", "tau_s"}
        unexpected = {
            key for key in (set(cfg.params) - allowed) if cfg.params.get(key) is not None
        }
        if unexpected:
            raise TypeError(
                f"running_leg_alternation received unsupported parameters: {sorted(unexpected)}"
            )
        super().__init__(
            cfg,
            env,
            label="running_leg_alternation",
            default_cap=50.0,
            cap_key="swing_cap_deg",
        )
        self._cap = math.radians(self._cap)
        self._entity, self._joint_ids = _asset_selection(cfg, env, term="running_leg_alternation")
        if self._joint_ids.size != 2:
            raise ValueError(
                "running_leg_alternation expects [left_hip_pitch, right_hip_pitch]"
            )
        self._sign = np.asarray([_LEFT_RIGHT_FORWARD_SIGN], dtype=get_global_dtype())

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        forward = self._entity.data.joint_pos[:, self._joint_ids] * self._sign
        return self._score(env, forward[:, 0] - forward[:, 1])


class running_stride_length(_SwingOscillationTerm):
    """Reward the feet scissoring past each other, measured at the ankles.

    The signal is the signed body-forward gap between the left and right ankle.
    Its swing is the stride: the feet must cross, so the gap changes sign every
    step. The previous version scored the gap's magnitude, which a static split
    stance maximized for free while the hips never moved.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        allowed = {"asset_cfg", "command_name", "command_threshold", "stride_cap", "tau_s"}
        unexpected = {
            key for key in (set(cfg.params) - allowed) if cfg.params.get(key) is not None
        }
        if unexpected:
            raise TypeError(
                f"running_stride_length received unsupported parameters: {sorted(unexpected)}"
            )
        super().__init__(
            cfg,
            env,
            label="running_stride_length",
            default_cap=0.10,
            cap_key="stride_cap",
        )
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("running_stride_length asset_cfg must be SceneEntityCfg")
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        body_ids = np.asarray(
            np.arange(self._entity.num_bodies, dtype=np.intp)[asset_cfg.body_ids]
        )
        if body_ids.size != 2:
            raise ValueError("running_stride_length asset_cfg must select [ankle_left, ankle_right]")
        self._foot_ids = body_ids

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        feet_w = np.asarray(self._entity.data.body_link_pos_w[:, self._foot_ids, :])
        root = np.asarray(self._entity.data.root_link_pos_w)
        quat = np.asarray(self._entity.data.root_link_quat_w)
        left_b = np_quat_apply_inverse(quat, feet_w[:, 0, :] - root)
        right_b = np_quat_apply_inverse(quat, feet_w[:, 1, :] - root)
        return self._score(env, left_b[:, 0] - right_b[:, 0])


class running_trunk_bow(ManagerTermBase):
    """Reward a forward trunk lean in ``[free_tilt_deg, max_tilt_deg]``.

    The sprint ``upright`` term fights the arched run the operator asked for.
    This term pays for a sagittal bow and is zero for a vertical trunk or a
    fall. Roll is ignored; heading terms own the yaw.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        allowed = {"asset_cfg", "free_tilt_deg", "max_tilt_deg"}
        unexpected = {
            key for key in (set(cfg.params) - allowed) if cfg.params.get(key) is not None
        }
        if unexpected:
            raise TypeError(
                f"running_trunk_bow received unsupported parameters: {sorted(unexpected)}"
            )
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("running_trunk_bow asset_cfg must be SceneEntityCfg")
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        free_tilt = _positive_real(
            cfg.params.get("free_tilt_deg", 12.0), label="running_trunk_bow free_tilt_deg"
        )
        max_tilt = _positive_real(
            cfg.params.get("max_tilt_deg", 40.0), label="running_trunk_bow max_tilt_deg"
        )
        if not free_tilt < max_tilt < 90.0:
            raise ValueError("running_trunk_bow requires free_tilt_deg < max_tilt_deg < 90")
        self._free = math.radians(free_tilt)
        self._max = math.radians(max_tilt)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del env, params
        quat = np.asarray(self._entity.data.root_link_quat_w)
        # Body-frame gravity x: positive when the trunk pitches forward.
        gravity_b = np_quat_apply_inverse(
            quat,
            np.broadcast_to(
                np.asarray([[0.0, 0.0, -1.0]], dtype=quat.dtype), (quat.shape[0], 3)
            ),
        )
        pitch = np.arcsin(np.clip(np.nan_to_num(gravity_b[:, 0], nan=0.0), -1.0, 1.0))
        ramp = np.clip(pitch / self._free, 0.0, 1.0)
        inside = (pitch >= 0.0) & (pitch <= self._max)
        return np.asarray(np.where(inside, ramp, 0.0), dtype=get_global_dtype())


class head_upright_hold(ManagerTermBase):
    """Keep the top of the head pointing at the sky.

    The head body's local ``up_axis_b`` is rotated into the world frame and
    compared with world up. Tilts within ``free_tilt_deg`` cost nothing, so the
    bird may still nod forward and reach while sprinting, and the reward then
    decays linearly to zero at ``zero_tilt_deg``. The linear shape is
    deliberate: the sprint policy folds ``neck_pitch`` onto its -90 deg stop and
    flips the head over, and a Gaussian bowl carries no gradient out of that
    saturated region.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        allowed = {"asset_cfg", "up_axis_b", "free_tilt_deg", "zero_tilt_deg"}
        unexpected = {
            key for key in (set(cfg.params) - allowed) if cfg.params.get(key) is not None
        }
        if unexpected:
            raise TypeError(
                f"head_upright_hold received unsupported parameters: {sorted(unexpected)}"
            )
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_HEAD_BODY_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("head_upright_hold asset_cfg must be SceneEntityCfg")
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        body_ids = np.asarray(
            np.arange(self._entity.num_bodies, dtype=np.intp)[asset_cfg.body_ids]
        )
        if body_ids.size != 1:
            raise ValueError("head_upright_hold asset_cfg must select exactly one body")
        self._body_id = int(body_ids[0])
        axis = np.asarray(cfg.params.get("up_axis_b", _DEFAULT_HEAD_UP_AXIS_B), dtype=np.float64)
        if axis.shape != (3,) or not np.isfinite(axis).all():
            raise ValueError("head_upright_hold up_axis_b must be three finite numbers")
        norm = float(np.linalg.norm(axis))
        if norm <= 0.0:
            raise ValueError("head_upright_hold up_axis_b must be nonzero")
        self._axis = np.asarray((axis / norm)[None, :], dtype=get_global_dtype())
        free_tilt = _positive_real(
            cfg.params.get("free_tilt_deg", 45.0), label="head_upright_hold free_tilt_deg"
        )
        zero_tilt = _positive_real(
            cfg.params.get("zero_tilt_deg", 120.0), label="head_upright_hold zero_tilt_deg"
        )
        if not free_tilt < zero_tilt <= 180.0:
            raise ValueError(
                "head_upright_hold requires free_tilt_deg < zero_tilt_deg <= 180"
            )
        self._free_cos = math.cos(math.radians(free_tilt))
        self._zero_cos = math.cos(math.radians(zero_tilt))

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del env, params
        quat = np.asarray(self._entity.data.body_link_quat_w[:, self._body_id, :])
        head_up_w = np_quat_apply(quat, self._axis)
        cos_tilt = np.nan_to_num(head_up_w[:, 2], nan=self._zero_cos)
        reward = (cos_tilt - self._zero_cos) / (self._free_cos - self._zero_cos)
        return np.asarray(np.clip(reward, 0.0, 1.0), dtype=get_global_dtype())


class running_spawn_progress(ManagerTermBase):
    """Reward capped world-frame speed along each environment's spawn heading."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        allowed = {"speed_cap", "asset_cfg"}
        unexpected = set(cfg.params) - allowed
        if unexpected:
            raise TypeError(
                f"running_spawn_progress received unsupported parameters: {sorted(unexpected)}"
            )
        self._cap = _positive_real(
            cfg.params.get("speed_cap", 1.2),
            label="running_spawn_progress speed_cap",
        )
        self._entity, self._heading_ref = _bind_heading_asset(
            cfg, env, label="running_spawn_progress"
        )

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._heading_ref[ids] = self._entity.data.heading_w[ids]

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        heading, _ = _refresh_heading_ref(env, self._entity, self._heading_ref)
        velocity = np.nan_to_num(self._entity.data.root_link_lin_vel_w, nan=0.0)
        along_heading = velocity[:, 0] * np.cos(self._heading_ref) + velocity[:, 1] * np.sin(
            self._heading_ref
        )
        del heading
        return np.asarray(
            np.clip(along_heading, 0.0, self._cap) / self._cap,
            dtype=get_global_dtype(),
        )


class running_command_ranges_curriculum:
    """Advance a forward-only speed band while preserving the command type."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        command_name = cfg.params.get("command_name")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError("running speed curriculum command_name must be non-empty")
        term_cfg = env.command_manager.get_term_cfg(command_name)
        if not isinstance(term_cfg, UniformVelocityCommandCfg):
            raise TypeError(
                "running speed curriculum requires UniformVelocityCommandCfg, "
                f"received {type(term_cfg).__name__}"
            )
        raw_stages = cfg.params.get("speed_stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("running speed curriculum speed_stages must be a non-empty list")
        stages: list[tuple[int, float, float]] = []
        previous_step = -1
        for index, stage in enumerate(raw_stages):
            if not isinstance(stage, dict) or set(stage) != {"step", "min_speed", "max_speed"}:
                raise ValueError(
                    "running speed curriculum stage "
                    f"{index} must contain step, min_speed, and max_speed"
                )
            step = stage["step"]
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or step < 0
                or step < previous_step
            ):
                raise ValueError("running speed curriculum steps must be nonnegative and ordered")
            minimum = float(stage["min_speed"])
            maximum = float(stage["max_speed"])
            if not math.isfinite(minimum) or not math.isfinite(maximum):
                raise ValueError("running speed curriculum bounds must be finite")
            if minimum < 0.0 or minimum > maximum:
                raise ValueError(
                    f"running speed curriculum has invalid speed band {(minimum, maximum)}"
                )
            stages.append((step, minimum, maximum))
            previous_step = step
        self._term_cfg = term_cfg
        self._stages = tuple(stages)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | slice,
        command_name: str,
        speed_stages: list[dict[str, Any]],
    ) -> dict[str, float]:
        del env_ids, command_name, speed_stages
        minimum, maximum = self._stages[0][1:]
        for step, stage_minimum, stage_maximum in self._stages:
            if env.common_step_counter >= step:
                minimum, maximum = stage_minimum, stage_maximum
        self._term_cfg.ranges.lin_vel_x = (minimum, maximum)
        return {"min_speed": minimum, "max_speed": maximum}


__all__ = [
    "head_facing_hold",
    "heading_abs_cost",
    "heading_hold",
    "running_command_ranges_curriculum",
    "running_forward_progress",
    "running_planar_drift_cost",
    "running_spawn_progress",
]
