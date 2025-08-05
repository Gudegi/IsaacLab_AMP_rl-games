from __future__ import annotations

import os
from dataclasses import MISSING

from assets.robots.amp_humanoid2 import AMP_HUMANOID

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg


@configclass
class AmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""
    
    # env
    episode_length_s = 10.0 # episode_length_s = dt * decimation * num_steps, EX) (1/120) * 2 * 300 = 5.0 sec
    decimation = 2 # controlFrequencyInv

    # 13 + 52 + 28 + 12 # [13(root_h1, root_rot6, root_vel3, root_ang_vel3), 52(dof_pos), 28(dof_vel), 12(key_body_pos)]
    observation_space = 105 
    action_space = 28
    state_space = 0
    num_amp_obs_steps = 2 # disc input dim = 210 (105 * 2)
    amp_observation_space = 105
    
    # bfs order body : 
    # ['pelvis(0)', 'torso(1)', 'right_thigh(2)', 'left_thigh(3)', 'head(4)', 'right_upper_arm(5)', 'left_upper_arm(6)', 
    # 'right_shin(7)', 'left_shin(8)', 'right_lower_arm(9)', 'left_lower_arm(10)', 'right_foot(11)', 'left_foot(12)', 
    # 'right_hand(13)', 'left_hand(14)', 'right_toe(15)', 'left_toe(16)'] -> no joint
    dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # character's joint index list except for root(0), right and left hand doesn't have joint.
    dof_offsets = [0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 25, 28] # joint_dof corresponding dof_body_ids. torso [0:3], ... left_ankle[25:28]
    dof_obs_size = 52 # 8*6 + 4*1
    dt = 1 / 60
    local_root_obs = True

    early_termination = True
    termination_height = 0.5

    root_body_name = "pelvis"
    key_body_names = ["left_foot", "right_foot", "left_hand", "right_hand"]
    state_init = "Hybrid" # Default, Random, Hybrid
    hybrid_init_prob = 1.0
    
    rel_motion_root_path = "../../../../assets/motions"
    motion_file = "amp_humanoid_run.npy"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=decimation,
        use_fabric=True,
        enable_scene_query_support=False,
        gravity=(0.0, 0.0, -9.81),
        
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0
        ),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
            enable_stabilization=True,

            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=5 * 2**15,
            gpu_found_lost_pairs_capacity=2**23,
            gpu_found_lost_aggregate_pairs_capacity=2**25,
            gpu_total_aggregate_pairs_capacity=2**21,
            gpu_heap_capacity=2**26,
            gpu_temp_buffer_capacity=2**24,
            gpu_max_num_partitions=8,
            gpu_max_soft_body_contacts=2**20,
            gpu_max_particle_contacts=2**20,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = AMP_HUMANOID.replace(prim_path="/World/envs/env_.*/Robot").replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                stiffness=None,
                damping=None,
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
        ),
    )