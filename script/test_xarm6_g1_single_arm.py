#!/usr/bin/env python3
"""Acceptance smoke test for the explicit single-arm xArm6-G1 mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.lift_pot_single_arm import lift_pot_single_arm  # noqa: E402


def resolve_args(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as stream:
        args = yaml.safe_load(stream)
    embodiment_cfg_path = ROOT / "task_config" / "_embodiment_config.yml"
    with embodiment_cfg_path.open(encoding="utf-8") as stream:
        embodiment_types = yaml.safe_load(stream)

    embodiment = args["embodiment"]
    if not args.get("single_arm", False) or len(embodiment) != 1:
        raise AssertionError("single-arm config must set single_arm=true and contain one embodiment")
    robot_file = (ROOT / embodiment_types[embodiment[0]]["file_path"]).resolve()
    args.update(
        {
            "task_name": "lift_pot_single_arm",
            "task_config": config_path.stem,
            "left_robot_file": str(robot_file),
            "right_robot_file": str(robot_file),
            "left_embodiment_config": yaml.safe_load(
                (robot_file / "config.yml").open(encoding="utf-8")
            ),
            "right_embodiment_config": yaml.safe_load(
                (robot_file / "config.yml").open(encoding="utf-8")
            ),
            "dual_arm_embodied": False,
            "embodiment_dis": 0.0,
            "need_plan": True,
            "save_data": False,
            "eval_mode": False,
            "eval_video_save_dir": None,
        }
    )
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "task_config" / "demo_clean_xarm6_g1_single.yml",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--active-arm", choices=("left", "right"), default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--check-action", action="store_true", help="exercise the 7D qpos policy path")
    parser.add_argument(
        "--check-modalities",
        action="store_true",
        help="also request depth and segmentation from the single wrist camera",
    )
    args_cli = parser.parse_args()

    args = resolve_args(args_cli.config.resolve())
    if args_cli.active_arm is not None:
        args["active_arm"] = args_cli.active_arm
    if args_cli.render:
        args["render_freq"] = 1
    if args_cli.check_modalities:
        args["data_type"]["depth"] = True
        args["data_type"]["mesh_segmentation"] = True
        args["data_type"]["actor_segmentation"] = True

    task = lift_pot_single_arm()
    task.setup_demo(now_ep_num=0, seed=args_cli.seed, **args)
    try:
        active = task.robot.get_active_entity()
        articulations = task.scene.get_all_articulations()
        # The kitchen pot is itself a movable articulation.  Count robot
        # articulations by their xArm joint signature rather than mistaking
        # that task object for a second robot.
        robot_articulations = [
            articulation
            for articulation in articulations
            if any(joint.get_name() == "joint1" for joint in articulation.get_active_joints())
        ]
        if len(robot_articulations) != 1:
            raise AssertionError(
                f"expected one xArm articulation, got {len(robot_articulations)} "
                f"(scene articulations={len(articulations)})"
            )
        inactive = task.robot.right_entity if task.active_arm == "left" else task.robot.left_entity
        if inactive is not None:
            raise AssertionError("inactive arm unexpectedly has a physical entity")
        if active is not robot_articulations[0]:
            raise AssertionError("active entity is not the only xArm articulation")

        active_joints = active.get_active_joints()
        policy_state = task.robot.get_active_arm_jointState()
        if len(active_joints) != 12:
            raise AssertionError(f"expected 12 active joints, got {len(active_joints)}")
        if len(policy_state) != 7:
            raise AssertionError(f"expected 7 policy values, got {len(policy_state)}")
        if task.robot.left_planner is None and task.robot.right_planner is None:
            raise AssertionError("single-arm CuRobo planner was not initialized")
        if task.robot.left_planner is not None and task.robot.right_planner is not None:
            raise AssertionError("single-arm mode initialized two planners")

        observation = task.get_obs()
        vector = np.asarray(observation["joint_action"]["vector"])
        if vector.shape != (7,):
            raise AssertionError(f"single-arm observation vector has shape {vector.shape}, expected (7,)")
        wrist_keys = {key for key in observation["observation"] if key.endswith("_camera")}
        expected_wrist_key = f"{task.active_arm}_camera"
        if expected_wrist_key not in wrist_keys or len(
            wrist_keys.intersection({"left_camera", "right_camera"})
        ) != 1:
            raise AssertionError(f"unexpected wrist camera keys: {sorted(wrist_keys)}")
        if args_cli.check_modalities:
            expected_camera_keys = {"head_camera", expected_wrist_key}
            if set(observation["observation"]) != expected_camera_keys:
                raise AssertionError(
                    "single-arm camera set changed while checking modalities: "
                    f"{sorted(observation['observation'])}"
                )
            for camera_name in expected_camera_keys:
                camera_obs = observation["observation"][camera_name]
                for field in ("rgb", "depth", "mesh_segmentation", "actor_segmentation"):
                    if field not in camera_obs:
                        raise AssertionError(f"{camera_name} is missing modality {field}")

        if args_cli.check_action:
            # Apply a small joint perturbation and an intermediate gripper
            # value, exercising actual 7D interpolation rather than only a
            # zero-length hold.
            probe_action = np.asarray(policy_state, dtype=np.float64).copy()
            probe_action[0] += 0.02
            probe_action[-1] = 0.8
            task.take_action(probe_action, action_type="qpos")
            if not task.last_action_plan_success:
                raise AssertionError("7D qpos probe required the planning fallback")
            if len(task.robot.get_active_arm_jointState()) != 7:
                raise AssertionError("7D qpos action changed the active state contract")

        task.play_once()
        success = bool(task.plan_success and task.check_success())
        pot_z = float(task.pot.get_pose().p[2])
        distance = float(
            np.linalg.norm(
                np.asarray(task.robot.get_active_tcp_pose()[:3])
                - np.asarray(task.pot.get_contact_point(task.contact_point_id)[:3])
            )
        )
        if not success:
            raise AssertionError(
                f"single-arm lift failed: plan_success={task.plan_success}, pot_z={pot_z:.4f}, "
                f"tcp_contact_distance={distance:.4f}"
            )
        print("PASS: one xArm articulation, one planner, 12 active joints, 7D observation, single-arm lift")
        print(
            f"active_arm={task.active_arm} xarm_articulations={len(robot_articulations)} "
            f"scene_articulations={len(articulations)} "
            f"active_joints={len(active_joints)} policy_dim={len(policy_state)} "
            f"pot_z={pot_z:.4f} tcp_contact_distance={distance:.4f}"
        )
    finally:
        task.close_env(clear_cache=False)


if __name__ == "__main__":
    main()
