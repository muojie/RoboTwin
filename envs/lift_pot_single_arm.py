"""Single-arm variant of the kitchen-pot lift smoke task.

The stock :mod:`envs.lift_pot` task is intentionally bilateral: it requires
two simultaneous grasps and therefore is not a valid single-arm acceptance
test.  This task keeps the same object and success thresholds, but uses one
configured arm and the corresponding pot handle only.
"""

from ._base_task import Base_Task
from .utils import *
import sapien
import math


class lift_pot_single_arm(Base_Task):

    def setup_demo(self, is_test=False, **kwargs):
        # The task config is expected to set single_arm=true.  Fail early if
        # somebody accidentally runs this task through the bilateral path.
        if not kwargs.get("single_arm", False):
            raise ValueError("lift_pot_single_arm requires single_arm: true")
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.model_name = "060_kitchenpot"
        self.model_id = np.random.randint(0, 2)
        self.arm_tag = ArmTag(self.active_arm)
        self.contact_point_id = 0 if self.active_arm == "left" else 1
        self.pot = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-0.05, 0.05],
            ylim=[-0.05, 0.05],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 8],
            qpos=[0.704141, 0, 0, 0.71006],
        )
        x, y = self.pot.get_pose().p[0], self.pot.get_pose().p[1]
        self.prohibited_area.append([x - 0.3, y - 0.1, x + 0.3, y + 0.1])

    def play_once(self):
        # One arm: close, grasp its handle, then lift vertically.
        self.move(self.close_gripper(self.arm_tag, pos=0.5))
        self.move(
            self.grasp_actor(
                self.pot,
                self.arm_tag,
                pre_grasp_dis=0.035,
                contact_point_id=self.contact_point_id,
            )
        )
        self.move(
            self.move_by_displacement(
                self.arm_tag,
                z=0.88 - self.pot.get_pose().p[2],
            )
        )

        self.info["info"] = {
            "{A}": f"{self.model_name}/base{self.model_id}",
            "{a}": str(self.arm_tag),
            "{contact_point_id}": str(self.contact_point_id),
        }
        return self.info

    def check_success(self):
        pot_pose = self.pot.get_pose()
        end = np.array(self.robot.get_active_tcp_pose()[:3])
        grasp = np.array(self.pot.get_contact_point(self.contact_point_id)[:3])
        pot_dir = get_face_prod(pot_pose.q, [0, 0, 1], [0, 0, 1])
        distance = np.sqrt(np.sum((end - grasp) ** 2))
        return (
            pot_pose.p[2] > 0.82
            and distance < 0.03
            and pot_dir > 0.8
            and self.robot.is_active_gripper_close()
        )
