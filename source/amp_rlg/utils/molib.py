# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Reference : IsaacGymEnvs and CALM motion_lib.py
# https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/amp/utils_amp/motion_lib.py
# https://github.com/NVlabs/CALM/blob/main/calm/utils/motion_lib.py

import numpy as np
import os
import yaml
import torch
from torch import nn

from amp_rlg.poselib.poselib.skeleton.my_skeleton3d import SkeletonMotion
from amp_rlg.poselib.poselib.core.my_rotation3d import *

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

@torch.jit.script
def normalize_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def quat_to_angle_axis(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qw, qx, qy, qz = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qz+1] / sin_theta_expand

    mask = sin_theta > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis

@torch.jit.script
def angle_axis_to_exp_map(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map

@torch.jit.script
def quat_to_exp_map(q: torch.Tensor) -> torch.Tensor:
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map

@torch.jit.script
def slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = 0, 1, 2, 3

    cos_half_theta = q0[..., qw] * q1[..., qw] \
                   + q0[..., qx] * q1[..., qx] \
                   + q0[..., qy] * q1[..., qy] \
                   + q0[..., qz] * q1[..., qz]
    
    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta);
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta; 
    
    new_q_w = ratioA * q0[..., qw:qw+1] + ratioB * q1[..., qw:qw+1]
    new_q_x = ratioA * q0[..., qx:qx+1] + ratioB * q1[..., qx:qx+1]
    new_q_y = ratioA * q0[..., qy:qy+1] + ratioB * q1[..., qy:qy+1]
    new_q_z = ratioA * q0[..., qz:qz+1] + ratioB * q1[..., qz:qz+1]

    cat_dim = len(new_q_w.shape) - 1
    new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=cat_dim)

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q


class MotionLib(nn.Module):
    def __init__(self, motion_file, num_dofs, key_body_ids, dof_body_ids, dof_offsets, device):
        super().__init__()
        self._num_dof = num_dofs
        self._key_body_ids = torch.tensor(key_body_ids, device=device)
        self._dof_body_ids = dof_body_ids # DOF_BODY_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] , character's joint index list except for root(0) -> total # = num_joint -1
        self._dof_offsets = dof_offsets # DOF_OFFSETS = [0, 3, 6, 9, 10, 11, 14, 17, 20, 23, 26, 27, 28] # lefthip[0:3], ... RightElbow[27:28]
        self._device = device
        self._load_motions(motion_file)

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)
        
        self.register_buffer(
            "global_translation", 
            torch.cat([m.global_translation for m in self._motions], dim=0).to(
                dtype=torch.float32, device=self._device
            ),
            persistent=False,
        )
        self.register_buffer(
            "global_rotation", 
            torch.cat([m.global_rotation for m in self._motions], dim=0).to(
                dtype=torch.float32, device=self._device
            ),
            persistent=False,
        )
        self.register_buffer(
            "local_rotation", 
            torch.cat([m.local_rotation for m in self._motions], dim=0).to(
                dtype=torch.float32, device=self._device
            ),
            persistent=False,
        )
        self.register_buffer(
            "global_root_velocity", 
            torch.cat([m.global_root_velocity for m in self._motions], dim=0).to(
                dtype=torch.float32, device=self._device
            ),
            persistent=False,
        )
        self.register_buffer(
            "global_root_angular_velocity", 
            torch.cat([m.global_root_angular_velocity for m in self._motions], dim=0).to(
                dtype=torch.float32, device=self._device
            ),
            persistent=False,
        )
        self.register_buffer(
            "dof_vels", 
            torch.cat([m.dof_vels for m in self._motions], dim=0).to(
                dtype=torch.float32, device=self._device
            ),
            persistent=False,
        )

        # Because a tensor has values of every motion, 
        # the 'length_starts' specify stariting frame of each motion.
        lengths = self._motion_num_frames # [77, 30, 20]
        lengths_shifted = lengths.roll(1) # [20, 77, 30]
        lengths_shifted[0] = 0 # [0, 77, 30]
        self.register_buffer(
            "length_starts", lengths_shifted.cumsum(0), persistent=False
        ) # [0, 77, 107]
        
        self.to(device)
        return

    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_motions(self, n):
        motion_ids = torch.multinomial(
            self._motion_weights, num_samples=n, replacement=True
        )
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        phase = torch.rand(size=motion_ids.shape, device=self._device)

        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert(truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len

        return motion_time

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]
    
    def get_motion_state(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.global_translation[f0l, 0]
        root_pos1 = self.global_translation[f1l, 0]

        root_rot0 = self.global_rotation[f0l, 0]
        root_rot1 = self.global_rotation[f1l, 0]

        local_rot0 = self.local_rotation[f0l]
        local_rot1 = self.local_rotation[f1l]

        root_vel = self.global_root_velocity[f0l]

        root_ang_vel = self.global_root_angular_velocity[f0l]
        
        key_pos0 = self.global_translation[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)] # [n, num_key_body, 3]
        key_pos1 = self.global_translation[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]

        key_pos0 = key_pos0.reshape(-1, len(self._key_body_ids)*3) # [n, 12]
        key_pos1 = key_pos1.reshape(-1, len(self._key_body_ids)*3)

        dof_vel = self.dof_vels[f0l]

        blend = blend.unsqueeze(-1)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

        root_rot = slerp(root_rot0, root_rot1, blend)

        key_pos = (1.0 - blend) * key_pos0 + blend * key_pos1
        #blend_exp = blend.unsqueeze(-1)
        #key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        
        local_rot = slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof(local_rot)

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos

    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []

        total_len = 0.0

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            curr_motion : SkeletonMotion = SkeletonMotion.from_file(curr_file)
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1) 

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
 
            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            # buggy
            #curr_motion.tensor = curr_motion.tensor.to(self._device)
            #curr_motion._skeleton_tree._parent_indices = curr_motion._skeleton_tree._parent_indices.to(self._device)
            #curr_motion._skeleton_tree._local_translation = curr_motion._skeleton_tree._local_translation.to(self._device)
            #curr_motion._rotation = curr_motion._rotation.to(self._device)

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            
            curr_weight = motion_weights[f]
            self._motion_weights.append(curr_weight)
            self._motion_files.append(curr_file)

        self._motion_lengths = torch.tensor(self._motion_lengths, dtype=torch.float32, device=self._device)
        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()
        
        self._motion_fps = torch.tensor(self._motion_fps, dtype=torch.float32, device=self._device)
        self._motion_dt = torch.tensor(self._motion_dt, dtype=torch.float32, device=self._device)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            motion_weights = []

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config['motions']
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                assert(curr_weight >= 0)

                curr_file = os.path.join(dir_name, curr_file)
                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]

        return motion_files, motion_weights

    def _calc_frame_blend(self, time, len, num_frames, dt):
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _compute_motion_dof_vels(self, motion):
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            frame_dof_vel = frame_dof_vel
            dof_vels.append(frame_dof_vel)
        
        dof_vels.append(dof_vels[-1])
        dof_vels = np.array(dof_vels)
        dof_vels = torch.tensor(dof_vels)

        return dof_vels
    
    def _local_rotation_to_dof(self, local_rot): 
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_q = local_rot[:, body_id]
                joint_exp_map = quat_to_exp_map(joint_q)
                dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map 
            elif (joint_size == 1):
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                joint_theta = joint_theta * joint_axis[..., 1] # y axis(knee, elbow)

                joint_theta = normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        dof_vel = np.zeros([self._num_dof])

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1) 
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt 
        local_vel = local_vel.numpy()

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel
            elif (joint_size == 1):
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[1] # y axis(knee, elbow)
            else:
                print("Unsupported joint type")
                assert(False)

        return dof_vel
