# UFACTORY xArm6 + G1 for RoboTwin

This embodiment is generated from UFACTORY's official `xarm_ros2` Jazzy branch:

- upstream: <https://github.com/xArm-Developer/xarm_ros2>
- pinned commit: `3dc2b5e8294758d96b54b15fa5920d581b7cbb3d`
- parameters: `dof=6`, `robot_type=xarm`, `limited=false`,
  `add_gripper=true`, `gripper_version=G1`, `mesh_suffix=stl`
- upstream license: BSD 3-Clause; see `LICENSE` in this directory

Run the generator from the RoboTwin repository root:

```bash
python -m pip install -r script/assets/requirements-xarm6-g1.txt
python script/assets/generate_xarm6_g1.py /path/to/xarm_ros2 \
  assets/embodiments/xarm6-g1
python script/update_embodiment_config_path.py
```

Validate the articulation and one real RoboTwin expert episode from the
repository root:

```bash
python script/test_xarm6_embodiment.py --check-curobo
python script/test_xarm6_lift_pot.py --seed 0
```

The second command also requires RoboTwin's official object assets. It does not
write a dataset; add `--render` to open the interactive SAPIEN viewer.

The adapter removes ROS/Gazebo/transmission elements, rewrites package mesh
URIs to relative paths, and removes the five G1 `<mimic>` elements. SAPIEN 3
does not execute those mimic constraints, so RoboTwin explicitly sends the same
G1 scalar to `drive_joint` and the five follower joints. The policy-facing
action remains 7 values per arm (6 arm joints + 1 normalized gripper value),
although each SAPIEN articulation exposes 12 active joints.

PhysX self-contact is disabled separately for each xArm articulation. The
official arm and G1 collision meshes overlap at fixed interfaces and otherwise
produce large internal impulses in SAPIEN. CuRobo remains responsible for arm
self-collision avoidance, while the two independent arms still collide with
each other, the table, and task objects.

The official `link_tcp` remains at 0.172 m from the flange. RoboTwin's
`gripper_bias` is 0.14 m because task contact points must land near the center
of the G1 finger pads (roughly 0.108-0.175 m), rather than on their outer edge.

The empty fixed link named `camera` uses the mounting transform from UFACTORY's
official D435 xacro so RoboTwin can render a wrist view. It is a virtual camera
frame only; there is no camera visual or collision mesh.

The nominal two-arm task config uses a 0.8 m base center distance and nominal
UFACTORY kinematics. Before sim-to-real work, measure the real base transforms,
match the real end-effectors/cameras, and generate a per-arm kinematics suffix
from each connected xArm6. The two physical xArm6 bodies alone do not make the
virtual G1 grippers or D435 frames transferable to hardware.
