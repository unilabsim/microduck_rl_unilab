"""Run the three published Sprint robustification stages.

Each stage resumes the previous checkpoint, freezes the 1.65–2.20 m/s command
band, and widens push / CoM / initial tilt. Stage magnitudes match
Vottivott/microduck-playground's Running release notes.

Usage:
    uv run --no-sync scripts/train_sprint_robust.py \\
      --load-run 2026-09-06_21-51-28_mujoco \\
      --checkpoint 11749
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl_ppo" / "MicroduckSprintFlat"
RESULTS_PATH = REPO_ROOT / "results" / "sprint_robust.jsonl"
STAGE_C = {
    "name": "C",
    "iterations": 200,
    "push": 0.10,
    "trunk": 0.008,
    "head": 0.006,
    "tilt_deg": 2.0,
}
# Between B and C. Full C dropped survival on the first pass, so long
# continuation holds here until the 2.2 m/s battery recovers.
STAGE_HOLD = {
    "name": "Hold",
    "iterations": 400,
    "push": 0.08,
    "trunk": 0.0065,
    "head": 0.006,
    "tilt_deg": 1.75,
}
STAGES = (
    {"name": "A", "iterations": 100, "push": 0.03, "trunk": 0.003, "head": 0.003, "tilt_deg": 1.0},
    {"name": "B", "iterations": 150, "push": 0.06, "trunk": 0.005, "head": 0.005, "tilt_deg": 1.5},
    STAGE_C,
)


def _range(magnitude: float) -> str:
    return f"[{-magnitude},{magnitude}]"


def _stage_overrides(stage: dict[str, float | int | str], *, iterations: int) -> list[str]:
    tilt = math.radians(float(stage["tilt_deg"]))
    push = float(stage["push"])
    trunk = float(stage["trunk"])
    head = float(stage["head"])
    return [
        f"algo.max_iterations={iterations}",
        "algo.save_interval=50",
        f"env.events.push_robot.params.velocity_range.x={_range(push)}",
        f"env.events.push_robot.params.velocity_range.y={_range(push)}",
        f"env.events.base_com.params.com_range.x={_range(trunk)}",
        f"env.events.base_com.params.com_range.y={_range(trunk)}",
        f"env.events.base_com.params.com_range.z={_range(trunk)}",
        f"env.events.head_com.params.com_range.x={_range(head)}",
        f"env.events.head_com.params.com_range.y={_range(head)}",
        f"env.events.head_com.params.com_range.z={_range(head)}",
        f"env.events.reset_base.params.pose_range.roll={_range(tilt)}",
        f"env.events.reset_base.params.pose_range.pitch={_range(tilt)}",
    ]


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _run_dirs() -> list[Path]:
    if not LOG_ROOT.is_dir():
        return []
    return [path for path in LOG_ROOT.glob("*_mujoco") if path.is_dir()]


def _latest_checkpoint(run_dir: Path) -> Path:
    models = sorted(
        run_dir.glob("model_*.pt"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    if not models:
        raise SystemExit(f"no checkpoints in {run_dir}")
    return models[-1]


def _append_result(record: dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _iter_results() -> list[dict[str, Any]]:
    if not RESULTS_PATH.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _best_nominal() -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for record in _iter_results():
        if record.get("stress"):
            continue
        survival = record.get("survival_fraction")
        if survival is None:
            continue
        if best is None or float(survival) > float(best["survival_fraction"]):
            best = record
    return best


def _train_stage(
    *,
    stage: dict[str, float | int | str],
    label: str,
    iterations: int,
    load_run: str,
    checkpoint: str,
    num_envs: int,
    known_runs: set[Path],
) -> tuple[Path, Path]:
    print(f"=== robust stage {label} ===", flush=True)
    before = {path.resolve() for path in known_runs}
    started = time.time()
    _run(
        [
            "uv",
            "run",
            "--no-sync",
            "microduck-train",
            "--algo",
            "ppo",
            "--task",
            "microduck_sprint_robust_flat",
            "--sim",
            "mujoco",
            f"algo.num_envs={num_envs}",
            f"algo.load_run={load_run}",
            f"algo.checkpoint={checkpoint}",
            "training.no_play=true",
            *_stage_overrides(stage, iterations=iterations),
        ]
    )
    created = [
        path
        for path in _run_dirs()
        if path.resolve() not in before and path.stat().st_mtime >= started - 5.0
    ]
    run_dir = max(created or _run_dirs(), key=lambda path: path.stat().st_mtime)
    ckpt = _latest_checkpoint(run_dir)
    return run_dir, ckpt


def _eval_checkpoint(
    *,
    ckpt: Path,
    run_dir: Path,
    label: str,
    eval_envs: int,
    stress: bool,
) -> dict[str, Any]:
    suffix = "stress" if stress else "2p2"
    eval_out = run_dir / f"eval_{suffix}_{eval_envs}_{label}.json"
    command = [
        "uv",
        "run",
        "--no-sync",
        "scripts/eval_sprint_speed.py",
        str(ckpt),
        "--num-envs",
        str(eval_envs),
        "--speed",
        "2.2",
        "--warmup",
        "1",
        "--duration",
        "10",
        "--output",
        str(eval_out),
    ]
    if stress:
        command.append("--stress")
    _run(command)
    payload = json.loads(eval_out.read_text(encoding="utf-8"))
    record = {
        "label": label,
        "run_dir": run_dir.name,
        "checkpoint": ckpt.name,
        "stress": stress,
        "survival_fraction": payload.get("survival_fraction"),
        "body_forward_speed_m_s": payload.get("body_forward_speed_m_s"),
        "mean_absolute_heading_error_deg": payload.get("mean_absolute_heading_error_deg"),
        "eval_path": str(eval_out),
    }
    _append_result(record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-run", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--from-best-jsonl", action="store_true")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--eval-envs", type=int, default=512)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--stages", default="A,B,C")
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=0.0,
        help="Keep training after the named stages until this wall-clock budget.",
    )
    parser.add_argument("--extra-chunk", type=int, default=400)
    parser.add_argument("--target-survival", type=float, default=0.988)
    parser.add_argument("--hold-until-survival", type=float, default=0.94)
    parser.add_argument("--final-stress", action="store_true")
    args = parser.parse_args()

    wanted = {name.strip().upper() for name in args.stages.split(",") if name.strip()}
    load_run = args.load_run
    checkpoint = args.checkpoint
    if args.from_best_jsonl:
        best = _best_nominal()
        if best is None:
            raise SystemExit("no nominal eval records in results/sprint_robust.jsonl")
        load_run = str(best["run_dir"])
        checkpoint = str(best["checkpoint"])
        print(f"resuming best jsonl record {best['label']} {load_run}/{checkpoint}", flush=True)
    if not load_run or not checkpoint:
        raise SystemExit("need --load-run and --checkpoint, or --from-best-jsonl")
    known_runs = set(_run_dirs())
    deadline = time.time() + max(args.budget_hours, 0.0) * 3600.0
    latest_payload: dict[str, Any] | None = None

    for stage in STAGES:
        if stage["name"] not in wanted:
            continue
        run_dir, ckpt = _train_stage(
            stage=stage,
            label=str(stage["name"]),
            iterations=int(stage["iterations"]),
            load_run=load_run,
            checkpoint=checkpoint,
            num_envs=args.num_envs,
            known_runs=known_runs,
        )
        known_runs.add(run_dir)
        load_run = run_dir.name
        checkpoint = ckpt.name
        if args.skip_eval:
            continue
        latest_payload = _eval_checkpoint(
            ckpt=ckpt,
            run_dir=run_dir,
            label=str(stage["name"]),
            eval_envs=args.eval_envs,
            stress=False,
        )

    extra_index = 0
    survival = (
        float(latest_payload["survival_fraction"])
        if latest_payload is not None and latest_payload.get("survival_fraction") is not None
        else 0.0
    )
    if survival <= 0.0 and args.from_best_jsonl:
        best = _best_nominal()
        if best is not None:
            survival = float(best["survival_fraction"])
    while (
        args.budget_hours > 0.0
        and time.time() + 25.0 * 60.0 < deadline
        and survival < args.target_survival
    ):
        extra_index += 1
        extra_stage = STAGE_C if survival >= args.hold_until_survival else STAGE_HOLD
        label = f"{extra_stage['name']}plus{extra_index}"
        run_dir, ckpt = _train_stage(
            stage=extra_stage,
            label=label,
            iterations=int(args.extra_chunk),
            load_run=load_run,
            checkpoint=checkpoint,
            num_envs=args.num_envs,
            known_runs=known_runs,
        )
        known_runs.add(run_dir)
        load_run = run_dir.name
        checkpoint = ckpt.name
        if args.skip_eval:
            continue
        latest_payload = _eval_checkpoint(
            ckpt=ckpt,
            run_dir=run_dir,
            label=label,
            eval_envs=args.eval_envs,
            stress=False,
        )
        survival = float(latest_payload["survival_fraction"])

    if args.final_stress and not args.skip_eval:
        run_dir = LOG_ROOT / load_run
        ckpt = _latest_checkpoint(run_dir)
        _eval_checkpoint(
            ckpt=ckpt,
            run_dir=run_dir,
            label="final_stress",
            eval_envs=args.eval_envs,
            stress=True,
        )


if __name__ == "__main__":
    main()
