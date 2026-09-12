"""Repair the head posture of the published fast gait without losing the gait.

The 1.68 m/s checkpoint already alternates its thighs and scissors its feet;
its defect is a head that spends most of the run upside down. Two earlier
retrofits applied the full head contract at once and halved the speed, so this
driver runs a single continuous resume against the
``microduck_sprint_gaitfix_flat`` owner, whose head weights ramp in over about
three thousand iterations while the leg-alternation and stride guards hold the
gait. A single run matters: ``reward_curriculum`` keys off
``env.common_step_counter``, which restarts whenever a new run starts.

Checkpoints are scored as they appear, on a second GPU so training keeps the
first. Each record carries the speed battery and the joint-level gait
diagnostic, because speed and survival alone cannot tell an alternating stride
from a parked split stance.

Usage:
    uv run --no-sync scripts/train_sprint_head_repair.py --budget-hours 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_B_RUN = REPO_ROOT / "logs" / "rsl_rl_ppo" / "MicroduckSprintFlat" / "2026-09-07_06-49-13_mujoco"
STAGE_B_CHECKPOINT = "11997"
DEFAULT_RESULTS = REPO_ROOT / "results" / "sprint_head_repair.jsonl"


def _checkpoints(run_dir: Path) -> list[Path]:
    return sorted(
        run_dir.glob("model_*.pt"),
        key=lambda path: int(path.stem.split("_")[1]),
    )


def _newest_run(log_root: Path, *, after: float) -> Path | None:
    if not log_root.is_dir():
        return None
    candidates = [
        path
        for path in log_root.glob("*_mujoco")
        if path.is_dir() and path.stat().st_mtime >= after - 5.0
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _eval_env(gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _run_json(command: list[str], *, output: Path, gpu: int) -> dict[str, Any] | None:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_eval_env(gpu),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not output.is_file():
        print(completed.stdout[-2000:], flush=True)
        print(completed.stderr[-2000:], flush=True)
        return None
    return json.loads(output.read_text(encoding="utf-8"))


def _score_checkpoint(
    checkpoint: Path,
    *,
    speed: float,
    eval_envs: int,
    duration: float,
    gpu: int,
) -> dict[str, Any] | None:
    battery_out = checkpoint.with_name(f"battery_{checkpoint.stem}.json")
    gait_out = checkpoint.with_name(f"gait_{checkpoint.stem}.json")
    battery = _run_json(
        [
            "uv", "run", "--no-sync", "scripts/eval_sprint_speed.py", str(checkpoint),
            "--speed", str(speed),
            "--num-envs", str(eval_envs),
            "--warmup", "1",
            "--duration", str(duration),
            "--device", "cuda:0",
            "--output", str(battery_out),
        ],
        output=battery_out,
        gpu=gpu,
    )
    if battery is None:
        return None
    gait = _run_json(
        [
            "uv", "run", "--no-sync", "scripts/diagnose_gait.py", str(checkpoint),
            "--speed", str(speed),
            "--num-envs", "64",
            "--duration", "6",
            "--device", "cuda:0",
            "--output", str(gait_out),
        ],
        output=gait_out,
        gpu=gpu,
    )
    if gait is None:
        return None
    hip = gait["left_right_split"]["hip_pitch"]
    speed_block = battery.get("body_forward_speed_m_s") or {}
    return {
        "checkpoint": str(checkpoint),
        "iteration": int(checkpoint.stem.split("_")[1]),
        "speed_m_s": speed_block.get("mean"),
        "survival_fraction": battery.get("survival_fraction"),
        "head_inverted_time_fraction": battery.get("head_inverted_time_fraction"),
        "head_up_tilt_deg": battery.get("head_up_tilt_deg"),
        "head_yaw_abs_deg": battery.get("head_yaw_abs_deg"),
        "head_roll_abs_deg": battery.get("head_roll_abs_deg"),
        "heading_error_deg": battery.get("mean_absolute_heading_error_deg"),
        "hip_split_sign_flip_fraction": hip["sign_flip_fraction"],
        "hip_split_peak_to_peak_deg": hip["peak_to_peak_deg"],
        "hip_split_mean_abs_deg": hip["mean_abs_deg"],
        "foot_gap_peak_to_peak_m": gait["foot_forward_gap_m"]["peak_to_peak"],
        "trunk_height_m": gait["trunk_height_m"]["mean"],
        "head_yaw_signed_deg": gait.get("head_yaw_signed_deg"),
        "head_roll_signed_deg": gait.get("head_roll_signed_deg"),
        "heading_signed_deg": gait.get("heading_signed_deg"),
        "heading_left_fraction": gait.get("heading_left_fraction"),
    }


def _passes(record: dict[str, Any], args: argparse.Namespace) -> bool:
    values = (
        record.get("speed_m_s"),
        record.get("survival_fraction"),
        record.get("head_inverted_time_fraction"),
        record.get("hip_split_sign_flip_fraction"),
        record.get("foot_gap_peak_to_peak_m"),
    )
    if any(value is None for value in values):
        return False
    speed, survival, inverted, sign_flip, gap = (float(value) for value in values)  # type: ignore[arg-type]
    heading = record.get("heading_error_deg")
    yaw = record.get("head_yaw_abs_deg")
    roll = record.get("head_roll_abs_deg")
    if heading is not None and float(heading) > args.max_heading:
        return False
    if yaw is not None and float(yaw) > args.max_head_yaw:
        return False
    if roll is not None and float(roll) > args.max_head_roll:
        return False
    return (
        speed >= args.min_speed
        and survival >= args.min_survival
        and inverted <= args.max_inverted
        and sign_flip >= args.min_sign_flip
        and gap >= args.min_foot_gap
    )


def _score(record: dict[str, Any]) -> float:
    speed = float(record.get("speed_m_s") or 0.0)
    survival = float(record.get("survival_fraction") or 0.0)
    inverted = float(record.get("head_inverted_time_fraction") or 1.0)
    heading = float(record.get("heading_error_deg") or 90.0)
    yaw = float(record.get("head_yaw_abs_deg") or 90.0)
    roll = float(record.get("head_roll_abs_deg") or 90.0)
    heading_quality = max(0.0, 1.0 - heading / 90.0)
    yaw_quality = max(0.0, 1.0 - yaw / 45.0)
    roll_quality = max(0.0, 1.0 - roll / 25.0)
    return speed * survival * (1.0 - inverted) * heading_quality * yaw_quality * roll_quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="microduck_sprint_gaitfix_flat")
    parser.add_argument("--load-run", default=str(STAGE_B_RUN))
    parser.add_argument("--checkpoint", default=STAGE_B_CHECKPOINT)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--algo-log-name", default="rsl_rl_ppo_headfix")
    parser.add_argument("--eval-gpu", type=int, default=1)
    parser.add_argument("--eval-envs", type=int, default=256)
    parser.add_argument("--eval-speed", type=float, default=2.2)
    parser.add_argument("--eval-duration", type=float, default=8.0)
    parser.add_argument("--final-eval-envs", type=int, default=512)
    parser.add_argument("--budget-hours", type=float, default=8.0)
    parser.add_argument("--min-speed", type=float, default=1.30)
    parser.add_argument("--min-survival", type=float, default=0.80)
    parser.add_argument("--max-inverted", type=float, default=0.05)
    parser.add_argument("--min-sign-flip", type=float, default=0.35)
    parser.add_argument("--min-foot-gap", type=float, default=0.15)
    parser.add_argument("--max-heading", type=float, default=75.0)
    parser.add_argument("--max-head-yaw", type=float, default=20.0)
    parser.add_argument("--max-head-roll", type=float, default=25.5)
    parser.add_argument("--results-jsonl", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()

    results_path = Path(args.results_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    log_root = REPO_ROOT / "logs" / args.algo_log_name / "MicroduckSprintFlat"
    deadline = time.time() + max(args.budget_hours, 0.0) * 3600.0

    command = [
        "uv", "run", "--no-sync", "microduck-train",
        "--algo", "ppo",
        "--task", args.task,
        "--sim", "mujoco",
        f"algo.num_envs={args.num_envs}",
        f"algo.load_run={args.load_run}",
        f"algo.checkpoint={args.checkpoint}",
        f"algo.max_iterations={args.iterations}",
        f"algo.save_interval={args.save_interval}",
        f"algo.algo_log_name={args.algo_log_name}",
        "training.task_name=MicroduckSprintFlat",
        "training.no_play=true",
    ]
    print("+", " ".join(command), flush=True)
    started = time.time()
    training_log = REPO_ROOT / "logs" / f"head_repair_{int(started)}.log"
    with training_log.open("w", encoding="utf-8") as handle:
        trainer = subprocess.Popen(command, cwd=REPO_ROOT, stdout=handle, stderr=handle)
    print(f"training log: {training_log}", flush=True)

    scored: set[Path] = set()
    records: list[dict[str, Any]] = []
    run_dir: Path | None = None
    try:
        while True:
            alive = trainer.poll() is None
            if run_dir is None:
                run_dir = _newest_run(log_root, after=started)
            if run_dir is not None:
                pending = [path for path in _checkpoints(run_dir) if path not in scored]
                # The newest file may still be mid-write while training runs.
                if alive and pending:
                    pending = pending[:-1]
                for checkpoint in pending:
                    scored.add(checkpoint)
                    record = _score_checkpoint(
                        checkpoint,
                        speed=args.eval_speed,
                        eval_envs=args.eval_envs,
                        duration=args.eval_duration,
                        gpu=args.eval_gpu,
                    )
                    if record is None:
                        continue
                    record["passes_gate"] = _passes(record, args)
                    record["score"] = _score(record)
                    records.append(record)
                    with results_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(json.dumps(record, ensure_ascii=False), flush=True)
            if not alive:
                break
            if time.time() > deadline:
                print("budget exhausted, stopping training", flush=True)
                trainer.terminate()
                trainer.wait(timeout=300)
                break
            time.sleep(60)
    finally:
        if trainer.poll() is None:
            trainer.terminate()
            trainer.wait(timeout=300)

    passing = [record for record in records if record["passes_gate"]]
    pool = passing or records
    if not pool:
        raise SystemExit("no checkpoint was scored")
    best = max(pool, key=_score)
    print(
        f"best {'gated' if passing else 'ungated'} checkpoint: {best['checkpoint']}\n"
        f"{json.dumps(best, ensure_ascii=False, indent=2)}",
        flush=True,
    )
    final = _score_checkpoint(
        Path(best["checkpoint"]),
        speed=args.eval_speed,
        eval_envs=args.final_eval_envs,
        duration=10.0,
        gpu=args.eval_gpu,
    )
    if final is not None:
        final["passes_gate"] = _passes(final, args)
        final["score"] = _score(final)
        final["label"] = "final"
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(final, ensure_ascii=False) + "\n")
        print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
