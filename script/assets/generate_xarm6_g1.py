#!/usr/bin/env python3
"""Generate the pinned UFACTORY xArm6 + G1 asset used by RoboTwin."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import xacro
import xacro.substitution_args as substitution_args
from lxml import etree


EXPECTED_COMMIT = "3dc2b5e8294758d96b54b15fa5920d581b7cbb3d"
CAMERA_XYZ = "0.06746 -0.0175 0.0237"
CAMERA_RPY = "3.141592653589793 -1.5707963267948966 0"


def _parse_xacro(path: Path, mappings: dict[str, str]) -> etree._Element:
    document = xacro.process_file(str(path), mappings=mappings)
    return etree.fromstring(
        document.toxml().encode(), etree.XMLParser(remove_comments=True)
    )


def _clean_urdf(root: etree._Element, description_root: Path, output: Path) -> int:
    root.set("name", "xarm6_g1")

    for tag in ("ros2_control", "transmission", "gazebo"):
        for element in list(root.findall(tag)):
            root.remove(element)

    # SAPIEN 3 does not execute URDF mimic constraints. RoboTwin already
    # expands one normalized gripper command to the six G1 joints explicitly,
    # so keeping mimic tags only makes planner joint parsing inconsistent.
    for mimic in list(root.xpath(".//mimic")):
        mimic.getparent().remove(mimic)

    # The upstream G1 xacro currently emits a stray root-level literal `">`.
    # URDF has no meaningful root-level text, so remove all such text/tails.
    root.text = None
    for child in root:
        child.tail = None

    # RoboTwin attaches its wrist renderer directly to a link named `camera`.
    # This empty frame uses UFACTORY's official D435 mounting transform; the
    # camera body is intentionally omitted because it is not collision geometry.
    camera_link = etree.SubElement(root, "link", name="camera")
    camera_joint = etree.SubElement(root, "joint", name="camera_joint", type="fixed")
    etree.SubElement(camera_joint, "parent", link="link_eef")
    etree.SubElement(camera_joint, "child", link="camera")
    etree.SubElement(camera_joint, "origin", xyz=CAMERA_XYZ, rpy=CAMERA_RPY)

    package_prefix = "package://xarm_description/"
    mesh_sources: set[tuple[Path, Path]] = set()
    for mesh in root.xpath(".//mesh"):
        uri = mesh.get("filename")
        if not uri or not uri.startswith(package_prefix):
            raise RuntimeError(f"unexpected mesh URI: {uri!r}")
        relative = Path(uri.removeprefix(package_prefix))
        mesh.set("filename", relative.as_posix())
        mesh_sources.add((description_root / relative, output / relative))

    for src, dst in sorted(mesh_sources):
        if not src.is_file():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        # Upstream has inconsistent executable bits on data files. Asset
        # permissions are normalized so a clean regeneration is reproducible.
        dst.chmod(0o644)

    expected_joints = {
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "drive_joint",
        "left_finger_joint",
        "left_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "right_finger_joint",
        "right_inner_knuckle_joint",
        "joint_tcp",
        "camera_joint",
    }
    actual_joints = {joint.get("name") for joint in root.findall("joint")}
    missing = expected_joints - actual_joints
    if missing:
        raise RuntimeError(f"generated URDF is missing joints: {sorted(missing)}")
    if root.xpath(".//mimic"):
        raise RuntimeError("mimic tags remain in the SAPIEN adapter URDF")

    return len(mesh_sources)


def _write_xml(root: etree._Element, path: Path) -> None:
    etree.ElementTree(root).write(
        str(path),
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
    )
    path.chmod(0o644)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="xarm_ros2 repository root")
    parser.add_argument("output", type=Path, help="RoboTwin embodiment directory")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    description_root = source / "xarm_description"
    moveit_root = source / "xarm_moveit_config"

    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"expected {EXPECTED_COMMIT}, got {commit}")
    dirty_files = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty_files:
        raise RuntimeError(f"xarm_ros2 has modified tracked files:\n{dirty_files}")

    package_roots = {
        "xarm_description": description_root,
        "xarm_moveit_config": moveit_root,
    }
    original_find = substitution_args._eval_find

    def resolve_package(package: str) -> str:
        if package in package_roots:
            return str(package_roots[package])
        return original_find(package)

    substitution_args._eval_find = resolve_package
    try:
        urdf_root = _parse_xacro(
            description_root / "urdf" / "xarm_device.urdf.xacro",
            {
                "dof": "6",
                "robot_type": "xarm",
                "limited": "false",
                "add_gripper": "true",
                "gripper_version": "G1",
                "add_realsense_d435i": "false",
                "mesh_suffix": "stl",
            },
        )
        srdf_root = _parse_xacro(
            moveit_root / "srdf" / "xarm.srdf.xacro",
            {
                "dof": "6",
                "robot_type": "xarm",
                "add_gripper": "true",
            },
        )
    finally:
        substitution_args._eval_find = original_find

    output.mkdir(parents=True, exist_ok=True)
    mesh_count = _clean_urdf(urdf_root, description_root, output)
    expected_meshes = {
        (output / Path(mesh.get("filename"))).resolve()
        for mesh in urdf_root.xpath(".//mesh")
    }
    existing_meshes = {
        path.resolve() for path in (output / "meshes").rglob("*") if path.is_file()
    }
    if stale_meshes := existing_meshes - expected_meshes:
        stale = "\n".join(str(path) for path in sorted(stale_meshes))
        raise RuntimeError(f"stale mesh files remain in output:\n{stale}")
    srdf_root.set("name", "xarm6_g1")
    _write_xml(urdf_root, output / "xarm6_g1.urdf")
    _write_xml(srdf_root, output / "xarm6_g1.srdf")
    license_text = (source / "LICENSE").read_text(encoding="utf-8")
    normalized_license = "\n".join(line.rstrip() for line in license_text.splitlines()) + "\n"
    license_path = output / "LICENSE"
    license_path.write_text(normalized_license, encoding="utf-8")
    license_path.chmod(0o644)

    print(f"source_commit={commit}")
    print(f"output={output}")
    print(f"meshes={mesh_count}")


if __name__ == "__main__":
    main()
