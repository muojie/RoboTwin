#!/usr/bin/env python3
"""Run one no-data lift_pot expert episode with two xArm6 + G1 arms."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.lift_pot import lift_pot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    asset_dir = ROOT / "assets" / "embodiments" / "xarm6-g1"
    objects_index = ROOT / "assets" / "objects" / "objaverse" / "list.json"
    if not objects_index.is_file():
        raise SystemExit("RoboTwin object assets are missing; download the official asset bundle first")
    if not (asset_dir / "curobo.yml").is_file():
        raise SystemExit("curobo.yml is missing; run python script/update_embodiment_config_path.py")

    with (ROOT / "task_config" / "demo_clean_xarm6_g1.yml").open(encoding="utf-8") as stream:
        task_args = yaml.safe_load(stream)
    with (asset_dir / "config.yml").open(encoding="utf-8") as stream:
        embodiment = yaml.safe_load(stream)

    task_args.update(
        {
            "task_name": "lift_pot",
            "task_config": "demo_clean_xarm6_g1",
            "left_robot_file": str(asset_dir),
            "right_robot_file": str(asset_dir),
            "left_embodiment_config": embodiment,
            "right_embodiment_config": embodiment,
            "dual_arm_embodied": False,
            "embodiment_dis": 0.8,
            "embodiment_name": "xarm6-g1+xarm6-g1",
            "save_data": False,
            "collect_data": False,
            "eval_mode": False,
            "need_plan": True,
            "render_freq": 1 if args.render else 0,
            "save_path": "/tmp/robotwin-xarm6-lift-pot-smoke",
        }
    )

    task = lift_pot()
    setup_started = time.monotonic()
    try:
        task.setup_demo(now_ep_num=0, seed=args.seed, **task_args)
        setup_seconds = time.monotonic() - setup_started
        play_started = time.monotonic()
        task.play_once()
        play_seconds = time.monotonic() - play_started
        success = bool(task.plan_success and task.check_success())
        pot_z = float(task.pot.get_pose().p[2])
        print(f"setup_seconds={setup_seconds:.2f}")
        print(f"play_seconds={play_seconds:.2f}")
        print(f"plan_success={task.plan_success}")
        print(f"task_success={success}")
        print(f"pot_z={pot_z:.4f}")
    finally:
        task.close_env(clear_cache=True)

    if not success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
