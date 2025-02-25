from __future__ import annotations

from enum import Enum
import gymnasium as gym
import numpy as np
import torch
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
#from isaaclab.utils.math import quat_mul, quat_rotate
from amp_rlg.utils.rotations import quat_mul, calc_heading_quat_inv, quat_to_tan_norm, exp_map_to_quat, quat_rotate
from .amp_env_cfg import AmpEnvCfg
from amp_rlg.utils.molib import MotionLib

class AmpEnv(DirectRLEnv):
    cfg: AmpEnvCfg

    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg: AmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.state_init = self.StateInit[self.cfg.state_init]

        # To minimize search time
        self._local_root_obs = self.cfg.local_root_obs 
        self._dof_obs_size = self.cfg.dof_obs_size
        self._dof_offsets = self.cfg.dof_offsets

        self.dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        self.dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self._build_pd_action_offset_scale()

        self.root_body_idx = self.robot.body_names.index(self.cfg.root_body_name)
        self.key_body_ids = [self.robot.data.body_names.index(i) for i in self.cfg.key_body_names]
        #self.robot.data.joint_names

        motion_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg.rel_motion_root_path)
        motion_file_path = os.path.join(motion_root, self.cfg.motion_file)
        
        self._motion_lib = MotionLib(motion_file=motion_file_path, 
                                    num_dofs=self.robot.num_joints,
                                    key_body_ids=self.key_body_ids, 
                                    dof_body_ids=self.cfg.dof_body_ids,
                                    dof_offsets=self.cfg.dof_offsets,
                                    device=self.device)
        
        self._amp_obs_dim = self.cfg.num_amp_obs_steps * self.cfg.amp_observation_space
        self._amp_obs_space = gym.spaces.Box(np.ones(self._amp_obs_dim) * -np.Inf, np.ones(self._amp_obs_dim) * np.Inf)
        self._amp_obs_buf = torch.zeros((self.num_envs, self.cfg.num_amp_obs_steps, self.cfg.amp_observation_space), device=self.device, dtype=torch.float)
        self._amp_obs_demo_buf: torch.Tensor = None 

    @property
    def amp_observation_space(self):
        return self._amp_obs_space
    
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    ### pre_physics_step
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
    
    def _apply_action(self):
        pd_tar = self._action_to_pd_targets(self.actions)
        self.robot.set_joint_position_target(pd_tar)

    ### post_physics_step
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.root_body_idx, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        if self.state_init == self.StateInit.Default:
            root_state, joint_pos, joint_vel = self._reset_default_state_init(env_ids)
        elif self.state_init == self.StateInit.Random \
              or self.state_init == self.StateInit.Start:
            root_state, joint_pos, joint_vel = self._reset_ref_state_init(env_ids)
        elif self.state_init == self.StateInit.Hybrid:
            num_envs = env_ids.shape[0]
            ref_probs = torch.tensor([self.cfg.hybrid_init_prob] * num_envs, device=self.device)
            ref_init_mask = torch.bernoulli(ref_probs) == 1.0
    
            ref_reset_ids = env_ids[ref_init_mask]
            if (len(ref_reset_ids) > 0):
                ref_root_state, ref_dof_pos, ref_dof_vel = self._reset_ref_state_init(ref_reset_ids)
                self.robot.write_root_link_pose_to_sim(ref_root_state[:, :7], ref_reset_ids)
                self.robot.write_root_com_velocity_to_sim(ref_root_state[:, 7:], ref_reset_ids)
                self.robot.write_joint_state_to_sim(ref_dof_pos, ref_dof_vel, None, ref_reset_ids)
            default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
            if (len(default_reset_ids) > 0):
                default_root_state, default_dof_pos, default_dof_vel = self._reset_default_state_init(default_reset_ids)
                self.robot.write_root_link_pose_to_sim(default_root_state[:, :7], default_reset_ids)
                self.robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], default_reset_ids)
                self.robot.write_joint_state_to_sim(default_dof_pos, default_dof_vel, None, default_reset_ids)
            return
        else:
            raise ValueError(f"Unknown reset strategy: {self.state_init}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _get_observations(self) -> dict:    
        obs = self._get_character_obs()

        observations = {"policy": obs}
        
        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_obs_steps - 1)): ## history update
            self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        # build AMP observation
        self._amp_obs_buf[:, 0] = obs.clone() ## current update
        self.extras = {"amp_obs": self._amp_obs_buf.view(-1, self._amp_obs_dim)}

        return observations
    
    def _get_character_obs(self) -> torch.Tensor:
        obs = build_amp_observations(
            self.robot.data.body_pos_w[:, self.root_body_idx],
            self.robot.data.body_quat_w[:, self.root_body_idx],
            self.robot.data.body_lin_vel_w[:, self.root_body_idx],
            self.robot.data.body_ang_vel_w[:, self.root_body_idx],
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.key_body_ids],
            self._local_root_obs,
            self._dof_obs_size,
            self._dof_offsets
        )
        return obs
    
    ###########################################################################################

    def _reset_default_state_init(self, env_ids: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # TODO: To update AMP obs buffer's current value, we have to know world body position in default posture.
        return root_state, joint_pos, joint_vel
    
    def _reset_ref_state_init(self, env_ids: torch.Tensor | None)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)

        if (self.state_init == self.StateInit.Random
            or self.state_init == self.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self.state_init == self.StateInit.Start):
            motion_times = np.zeros(num_envs)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self.state_init))

        root_pos, root_quat, dof_pos, root_vel, root_ang_vel, dof_vel, _ \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = root_pos + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.15
        root_state[:, 3:7] = root_quat
        root_state[:, 7:10] = root_vel
        root_state[:, 10:13] = root_ang_vel

        # Update the current value of the AMP obs buffer. 
        # the history doesn't matter because it is updated anyway by _get_observations().
        self._init_amp_obs_ref(env_ids, motion_ids, motion_times)
        return root_state, dof_pos, dof_vel
    
    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        # prior pose of selected one
        dt = self.cfg.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self.cfg.num_amp_obs_steps - 1])        
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self.cfg.num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps
        
        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(
            root_pos, 
            root_rot, 
            root_vel, 
            root_ang_vel, 
            dof_pos, 
            dof_vel, 
            key_pos, 
            self._local_root_obs,
            self._dof_obs_size,
            self._dof_offsets)
        self._amp_obs_buf[env_ids, 0] = amp_obs_demo.view(self._amp_obs_buf[env_ids, 0].shape)
        return


#################### Called by training network ###################################################

    def fetch_amp_obs_demo(self, num_samples: int):
        """
            num_samples = self._amp_batch_size
        """
        dt = self.cfg.dt
        motion_ids = self._motion_lib.sample_motions(num_samples)

        if self._amp_obs_demo_buf is None:
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        ## since negative times are added to these values in build_amp_obs_demo,
        ## we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = dt * (self.cfg.num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time)
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self.cfg.num_amp_obs_steps])        
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self.cfg.num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(
            root_pos, 
            root_rot, 
            root_vel, 
            root_ang_vel, 
            dof_pos, 
            dof_vel, 
            key_pos, 
            self._local_root_obs, 
            self._dof_obs_size, 
            self._dof_offsets)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)

        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self._amp_obs_dim)
        return amp_obs_demo_flat
    
    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self.cfg.num_amp_obs_steps, self.cfg.amp_observation_space), device=self.device, dtype=torch.float)
        return
    
###################################################################################################

    def _build_pd_action_offset_scale(self):
        humanoid_num_joints = len(self._dof_offsets) - 1

        lim_low = self.dof_lower_limits#.cpu().numpy()
        lim_high = self.dof_upper_limits#.cpu().numpy()

        for j in range(humanoid_num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]

            if (dof_size == 3):
                lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = torch.tensor(self._pd_action_offset, device=self.device)
        self._pd_action_scale = torch.tensor(self._pd_action_scale, device=self.device)
        return
    
    def _action_to_pd_targets(self, action: torch.Tensor) -> torch.Tensor:
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar
    
###################################################################################################

@torch.jit.script 
def dof_to_obs(pose: torch.Tensor, dof_obs_size: int, dof_offsets: list[int]) -> torch.Tensor:
    #dof_obs_size = 52 # 8*6 + 4*1 
    #dof_offsets = [0, 3, 6, 9, 10, 11, 14, 17, 20, 23, 26, 27, 28]
    humanoid_num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(humanoid_num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset:(dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs

@torch.jit.script
def build_amp_observations(
    root_pos: torch.Tensor, 
    root_rot: torch.Tensor, 
    root_vel: torch.Tensor, 
    root_ang_vel: torch.Tensor, 
    dof_pos: torch.Tensor, 
    dof_vel: torch.Tensor, 
    key_body_pos: torch.Tensor, 
    local_root_obs: bool,
    dof_obs_size: int, 
    dof_offsets: list[int]
    ) -> torch.Tensor:

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    key_body_pos = key_body_pos.view(heading_rot.shape[0], -1) # (num_envs, 4, 3) -> (num_envs, 12)
    root_pos_expand = root_pos.repeat(1, int(key_body_pos.shape[1]/3))
    local_key_body_pos = key_body_pos - root_pos_expand
    local_key_pos = local_key_body_pos.clone()
    
    local_key_pos[:, 0:3] = quat_rotate(heading_rot[:], local_key_body_pos[:, 0:3])
    local_key_pos[:, 3:6] = quat_rotate(heading_rot[:], local_key_body_pos[:, 3:6])
    local_key_pos[:, 6:9] = quat_rotate(heading_rot[:], local_key_body_pos[:, 6:9])
    local_key_pos[:, 9:12] = quat_rotate(heading_rot, local_key_body_pos[:, 9:12])

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)

    # 1, 6, 3, 3, 52, 28, key_pos * 3
    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, local_key_pos), dim=-1)
    return obs