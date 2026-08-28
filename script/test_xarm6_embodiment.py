#!/usr/bin/env python3
"""Smoke-test the RoboTwin xArm6 + G1 embodiment without running a task."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import sapien.core as sapien
import yaml
from lxml import etree
from transforms3d.quaternions import quat2mat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import only envs.robot for this asset smoke test. The normal envs package
# imports every task utility eagerly and therefore requires the multi-GB object
# bundle, which is unrelated to validating a robot articulation.
envs_package = types.ModuleType("envs")
envs_package.__path__ = [str(ROOT / "envs")]
envs_package.__package__ = "envs"
sys.modules.setdefault("envs", envs_package)

utils_package = types.ModuleType("envs.utils")
utils_package.__path__ = [str(ROOT / "envs" / "utils")]
utils_package.__package__ = "envs.utils"
transforms_spec = importlib.util.spec_from_file_location(
    "envs.utils.transforms", ROOT / "envs" / "utils" / "transforms.py"
)
assert transforms_spec and transforms_spec.loader
transforms_module = importlib.util.module_from_spec(transforms_spec)
transforms_spec.loader.exec_module(transforms_module)
utils_package.transforms = transforms_module
sys.modules.setdefault("envs.utils", utils_package)
sys.modules.setdefault("envs.utils.transforms", transforms_module)

robot_package = types.ModuleType("envs.robot")
robot_package.__path__ = [str(ROOT / "envs" / "robot")]
robot_package.__package__ = "envs.robot"
sys.modules.setdefault("envs.robot", robot_package)

from envs.robot.robot import Robot  # noqa: E402
from envs.robot.planner import CuroboPlanner  # noqa: E402


ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
GRIPPER_JOINTS = [
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
]
REQUIRED_LINKS = {
    "link_base",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link_eef",
    "xarm_gripper_base_link",
    "link_tcp",
    "camera",
}


def validate_files(asset_dir: Path, config: dict) -> Path:
    urdf_path = asset_dir / config["urdf_path"]
    srdf_path = asset_dir / config["srdf_path"]
    if not urdf_path.is_file() or not srdf_path.is_file():
        raise AssertionError("URDF/SRDF is missing; run the xArm6 asset generator")

    root = etree.parse(str(urdf_path)).getroot()
    if root.xpath(".//mimic"):
        raise AssertionError("SAPIEN adapter URDF must not contain mimic tags")
    for tag in ("ros2_control", "transmission", "gazebo"):
        if root.findall(tag):
            raise AssertionError(f"unsupported URDF extension remains: {tag}")

    links = {link.get("name") for link in root.findall("link")}
    joints = {joint.get("name") for joint in root.findall("joint")}
    if missing := REQUIRED_LINKS - links:
        raise AssertionError(f"missing links: {sorted(missing)}")
    expected_joints = set(ARM_JOINTS + GRIPPER_JOINTS + ["joint_tcp", "camera_joint"])
    if missing := expected_joints - joints:
        raise AssertionError(f"missing joints: {sorted(missing)}")

    for mesh in root.xpath(".//mesh"):
        mesh_path = urdf_path.parent / mesh.get("filename")
        if not mesh_path.is_file():
            raise AssertionError(f"missing mesh: {mesh_path}")
    return urdf_path


def drive_targets(robot: Robot, arm_tag: str) -> dict[str, float]:
    joints = robot.left_gripper if arm_tag == "left" else robot.right_gripper
    return {joint.get_name(): float(joint.get_drive_target()[0]) for joint, _, _ in joints}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=ROOT / "assets" / "embodiments" / "xarm6-g1",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--check-curobo",
        action="store_true",
        help="initialize CuRobo and plan a 2 cm Cartesian motion on the GPU",
    )
    args = parser.parse_args()

    asset_dir = args.asset_dir.resolve()
    with (asset_dir / "config.yml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_files(asset_dir, config)

    engine = sapien.Engine()
    scene = engine.create_scene()
    scene.set_timestep(1 / 250)
    robot = Robot(
        scene,
        need_topp=False,
        left_embodiment_config=config,
        right_embodiment_config=config,
        left_robot_file=str(asset_dir),
        right_robot_file=str(asset_dir),
        dual_arm_embodied=False,
        embodiment_dis=0.8,
    )
    robot.init_joints()

    expected_active = set(ARM_JOINTS + GRIPPER_JOINTS)
    for side, entity in (("left", robot.left_entity), ("right", robot.right_entity)):
        active_joints = entity.get_active_joints()
        active_names = {joint.get_name() for joint in active_joints}
        if active_names != expected_active:
            raise AssertionError(f"{side} active joints differ: {sorted(active_names)}")
        qpos = entity.get_qpos()
        arm_qpos = [qpos[active_joints.index(entity.find_joint_by_name(name))] for name in ARM_JOINTS]
        home = config["homestate"][0 if side == "left" else 1]
        if not np.allclose(arm_qpos, home, atol=1e-6):
            raise AssertionError(f"{side} was not initialized at home: {arm_qpos}")
        gripper_qpos = [
            qpos[active_joints.index(entity.find_joint_by_name(name))] for name in GRIPPER_JOINTS
        ]
        if not np.allclose(gripper_qpos, 0.0, atol=1e-6):
            raise AssertionError(f"{side} gripper was not initialized open: {gripper_qpos}")

    robot.set_gripper(1.0, "left", gripper_eps=0)
    robot.set_gripper(1.0, "right", gripper_eps=0)
    for side in ("left", "right"):
        targets = drive_targets(robot, side)
        if set(targets) != set(GRIPPER_JOINTS) or not np.allclose(list(targets.values()), 0.0):
            raise AssertionError(f"{side} open targets are wrong: {targets}")

    for _ in range(args.steps):
        scene.step()

    robot.set_gripper(0.0, "left", gripper_eps=0)
    robot.set_gripper(0.0, "right", gripper_eps=0)
    for side in ("left", "right"):
        targets = drive_targets(robot, side)
        if not np.allclose(list(targets.values()), 0.85):
            raise AssertionError(f"{side} close targets are wrong: {targets}")

    for _ in range(args.steps):
        scene.step()

    left_root = robot.left_entity.get_root_pose().p
    right_root = robot.right_entity.get_root_pose().p
    if not np.isclose(right_root[0] - left_root[0], 0.8, atol=1e-5):
        raise AssertionError(f"base spacing is wrong: left={left_root}, right={right_root}")

    for side, entity in (("left", robot.left_entity), ("right", robot.right_entity)):
        if not np.isfinite(entity.get_qpos()).all() or not np.isfinite(entity.get_qvel()).all():
            raise AssertionError(f"{side} articulation became non-finite")
        active_joints = entity.get_active_joints()
        qpos = entity.get_qpos()
        arm_qpos = [qpos[active_joints.index(entity.find_joint_by_name(name))] for name in ARM_JOINTS]
        home = config["homestate"][0 if side == "left" else 1]
        if not np.allclose(arm_qpos, home, atol=0.05):
            raise AssertionError(f"{side} arm drifted away from home: {arm_qpos}")

        ignore_bit = 1 << (30 if side == "left" else 29)
        for link in entity.get_links():
            for shape in link.get_collision_shapes():
                if not shape.get_collision_groups()[2] & ignore_bit:
                    raise AssertionError(f"{side} self-collision filter is missing on {link.get_name()}")
        tcp = entity.find_link_by_name("link_tcp").get_pose()
        if tcp.p[2] <= 0.74:
            raise AssertionError(f"{side} TCP is below the table: {tcp.p}")

        joint6 = entity.find_joint_by_name("joint6").global_pose
        link6 = entity.find_link_by_name("link6").get_pose()
        fixed_rotation = quat2mat(joint6.q).T @ quat2mat(link6.q)
        if not np.allclose(fixed_rotation, np.asarray(config["global_trans_matrix"]), atol=2e-4):
            raise AssertionError(f"{side} joint6/link6 frame calibration differs")

        camera = entity.find_link_by_name("camera").get_pose()
        camera_forward = quat2mat(camera.q)[:, 0]
        tool_forward = quat2mat(link6.q)[:, 2]
        if float(camera_forward @ tool_forward) < 0.999:
            raise AssertionError(f"{side} virtual wrist camera does not face along the tool")

    if args.check_curobo:
        curobo_path = asset_dir / "curobo.yml"
        if not curobo_path.is_file():
            raise AssertionError("curobo.yml is missing; run update_embodiment_config_path.py")
        active_names = [joint.get_name() for joint in robot.left_entity.get_active_joints()]
        planner = CuroboPlanner(
            robot.left_entity.get_root_pose(),
            ARM_JOINTS,
            active_names,
            yml_path=str(curobo_path),
        )
        current = robot.left_entity.find_link_by_name("link6").get_pose()
        target = sapien.Pose(current.p + np.array([0.0, 0.02, 0.0]), current.q)
        result = planner.plan_path(robot.left_entity.get_qpos(), target, arms_tag="left")
        if result.get("status") != "Success":
            raise AssertionError(f"CuRobo plan failed: {result}")
        if not np.isfinite(result["position"]).all():
            raise AssertionError("CuRobo returned a non-finite trajectory")
        print(f"curobo_waypoints={result['position'].shape[0]}")

    print("PASS: official xArm6/G1 files, dual-arm load, 6+1 mapping, home, gripper, and camera")
    print(f"left_root={np.round(left_root, 4).tolist()} right_root={np.round(right_root, 4).tolist()}")
    print(f"active_per_arm={len(expected_active)} policy_values_per_arm=7")


if __name__ == "__main__":
    main()
