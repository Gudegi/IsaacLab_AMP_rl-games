from isaacsim import SimulationApp

config = {"headless": False,}

simulation_app = SimulationApp(config)

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.storage.native import get_assets_root_path

import carb
import numpy as np
import omni.appwindow
import torch
from pxr import Sdf, Gf, UsdLux
import os

from amp_rlg.utils.molib import MotionLib

ASSET_ROOT_PATH = "./assets"

class MotionViewer(object):
    def __init__(self, motion_path: str, physics_dt: float = 1.0/60.0, render_dt: float =1.0/60.0) -> None:
        """
        Argument:
        motion_path {str} -- Motion Path
        physics_dt {float} -- Physics downtime of the scene.
        render_dt {float} -- Render downtime of the scene.

        """
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.util.debug_draw")
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        prim = get_prim_at_path("/World/GroundPlane")
        if not prim.IsValid():
            prim = define_prim("/World/GroundPlane", "Xform")
            asset_path = assets_root_path + "/Isaac/Environments/Terrains/flat_plane.usd"
            #asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
            prim.GetReferences().AddReference(asset_path)

        self._create_distant_light()
        
        self._usd_path = ASSET_ROOT_PATH + "/robots/amp_humanoid2-instanceable.usd"
        add_reference_to_stage(usd_path=self._usd_path, prim_path="/World/amp_humanoid2")

        self._robot = self._world.scene.add(
            SingleArticulation(
                prim_path="/World/amp_humanoid2",
                name="amp_humanoid2",
                translation=np.array([[0, 0, 0.70]]),
            )
        )
        print("!!!!!!!!!!!!!!!!!!!!!!!!!")
        self._world.reset()
        self._robot.post_reset()
        self.needs_reset = False
        self._device = "cpu"

        self.key_body_ids = torch.tensor([11, 12, 13, 14]) # todo: hardcoded
        self.dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.dof_offsets = [0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 25, 28]
        
        self.motion_lib =  MotionLib(motion_file=motion_path, 
                               num_dofs=self._robot.num_dof,
                               key_body_ids=self.key_body_ids,
                               dof_body_ids=self.dof_body_ids,
                               dof_offsets=self.dof_offsets,
                               device=self._device)

        motion_dt = 1 / self.motion_lib._motion_fps
        self.total_frames = int(self.motion_lib._motion_num_frames[0])
        self.motion_time = self.motion_lib.get_motion_length(motion_ids=[0])
        print("Motion dt : ", motion_dt)
        print("Motion Frames : ", self.total_frames)
        print("Motion Time : ", self.motion_time)

        render_dt = self._world.get_rendering_dt()
        physics_dt = self._world.get_physics_dt()
        print("Rendering dt : ", render_dt)
        print("Physics dt : ", physics_dt)
        self.camera = Camera()

    def _create_distant_light(self, prim_path="/World/defaultDistantLight", intensity=5000):
            stage = get_current_stage()
            light = UsdLux.DistantLight.Define(stage, prim_path)
            light.CreateIntensityAttr().Set(intensity)
    
    def setup(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        #self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._world.add_physics_callback("motion_sync", callback_fn=self.on_physics_step)

    def on_physics_step(self, step_size) -> None:
        if self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
        world_time_step = self._world.current_time_step_index
        #clipped_frame_count = world_time_step
        clipped_frame_count = world_time_step % self.total_frames
        root_pos = self.motion_sync(torch.tensor([0]), clipped_frame_count, step_size)
        #self.camera._update_camera(root_pos[0])
        
    def run(self) -> None:
        while simulation_app.is_running():
            self._world.step(render=True)
            if not self._world.is_simulating():
                self.needs_reset = True
        return
    
    def motion_sync(self, motion_ids, frame_count, dt):
        motion_ids = motion_ids
        motion_times = frame_count * dt
        
        motion_times = torch.tensor(motion_times, device=self._device)
        motion_times = motion_times.flatten()
        
        if motion_times + dt > self.motion_time:
            self.needs_reset = True

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = self.motion_lib.get_motion_state(motion_ids, motion_times)

        self._robot.set_world_pose(root_pos, root_rot)
        self._robot.set_linear_velocity(root_vel)
        self._robot.set_angular_velocity(root_ang_vel)
        self._robot.set_joint_positions(dof_pos)
        self._robot.set_joint_velocities(dof_vel)

        self._draw.clear_points()
        self._draw.clear_lines()
        
        self._draw.draw_points(key_pos[:, 0:3].cpu().numpy(), [(1, 0, 0, 0.5)] * 1, [30] * 1)
        self._draw.draw_points(key_pos[:, 3:6].cpu().numpy(), [(0, 1, 0, 0.5)] * 1, [30] * 1)
        self._draw.draw_points(key_pos[:, 6:9].cpu().numpy(), [(0, 0, 1, 0.5)] * 1, [30] * 1)
        self._draw.draw_points(key_pos[:, 9:12].cpu().numpy(), [(1, 1, 0, 0.5)] * 1, [30] * 1)
        return root_pos
        

class Camera(object):
    def __init__(self):
        from omni.kit.viewport.utility import get_viewport_from_window_name
        stage = omni.usd.get_context().get_stage()
        self.view_port = get_viewport_from_window_name("Viewport")
        # Create camera
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.view_port.set_active_camera(self.perspective_path)

    def _update_camera(self, root_pos):
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        camera_local_transform = torch.tensor([-1.7, 0.0, 0.6])
        camera_pos = camera_local_transform + root_pos

        camera_state = ViewportCameraState(self.camera_path, self.view_port)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(root_pos[0].item(), root_pos[1].item(), root_pos[2].item()+0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)

        self.view_port.set_active_camera(self.camera_path)
        return


def main():

    sub_folder = "./motions"
    file_name = "amp_humanoid_jog.npy"
    #file_name = "amp_humanoid_walk.npy"
    #file_name = "amp_humanoid_run.npy"
    motion_path = os.path.join(ASSET_ROOT_PATH, sub_folder, file_name)

    runner = MotionViewer(motion_path, physics_dt=1.0/30., render_dt=1.0/120.0)
    simulation_app.update()
    runner.setup()

    runner._world.reset()
    runner.run()
    simulation_app.close()

if __name__ == "__main__":
    main()
