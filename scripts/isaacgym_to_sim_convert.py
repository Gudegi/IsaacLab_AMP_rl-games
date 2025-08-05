
import torch
import pathlib
from tqdm import tqdm
import numpy as np

from source.amp_rlg.poselib.poselib.core.my_rotation3d import *
from source.amp_rlg.poselib.poselib.skeleton.my_skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

# from isaac gym motion to isaac sim motion for AMP family motions.
# [x, y, z, w] -> [w, x, y, z], DFS -> BFS

def dfs_to_bfs_order(dfs_node_names, bfs_node_names):
    dfs_to_bfs_order_list = []
    for i, bfs_name in enumerate(bfs_node_names):
        for j, dfs_name in enumerate(dfs_node_names):
            if dfs_name == bfs_name:
                dfs_to_bfs_order_list.append(j) 
    return dfs_to_bfs_order_list

def parse_dof_npy(motion_file_path):
    motion_data = np.load(motion_file_path, allow_pickle=True).item()
    frame_rate = int(motion_data.get('fps')) # 30 
    
    #global_velocity = motion_data.get("global_velocity")['arr'].astype(motion_data.get("root_translation")['context']['dtype'])
    #global_angular_velocity = motion_data.get('global_angular_velocity')['arr'].astype(motion_data.get("root_translation")['context']['dtype'])
    if motion_data.get('is_local') == True:
        local_rotation = motion_data.get("rotation")['arr'].astype(np.float32)
    else:
        assert(False) # not supported yet.
    root_translation = motion_data.get("root_translation")['arr'].astype(np.float32)
    local_rotation = local_rotation[:, :, [3, 0, 1, 2]] # [x, y, z, w] -> [w, x, y, z]
    local_rotation = torch.from_numpy(local_rotation)
    root_translation = torch.from_numpy(root_translation)
    return frame_rate, local_rotation, root_translation

def export_one(skel_tree, motion_file_path, output_final_path, dfs_to_bfs_order_list, joint_ids:list=None, slice:tuple=None):
    try:
        frame_rate, local_rotation, root_translation = parse_dof_npy(motion_file_path)
        
        if joint_ids != None:
            tmp_local_rotation = torch.zeros((local_rotation.shape[0], len(skel_tree.node_names), 4), dtype=torch.float32)
            tmp_local_rotation[:, :, 0] = 1
            tmp_local_rotation[:, joint_ids, :] = local_rotation.clone()
            local_rotation = tmp_local_rotation.clone()
        local_rotation = local_rotation[:, dfs_to_bfs_order_list, :] ## DFS to BFS
        
        if slice:
            local_rotation = local_rotation[slice[0]:(slice[1]+1), :, :]
            root_translation = root_translation[slice[0]:(slice[1]+1), :]
        new_poses = SkeletonState.from_rotation_and_root_translation(skeleton_tree=skel_tree, r=local_rotation, t=root_translation, is_local=True)
        motion = SkeletonMotion.from_skeleton_state(new_poses, frame_rate)
        motion.to_file(output_final_path)
        return True
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        return False
    
def convert_one(skel_tree, in_file_path, out_file_path, dfs_to_bfs_list, joint_ids:list=None, slice:tuple=None):
    output = export_one(skel_tree, in_file_path, out_file_path, dfs_to_bfs_list, joint_ids, slice)
    if output == False:
        print('failed file, ', in_file_path, '!!!!!!!!!!!!!!!!!!!!!!!')
    pass

def convert_all(skel_tree, out_character_name, in_root_path, out_root_path, dfs_to_bfs_list, joint_ids:list=None):

    input_base_path = pathlib.Path.home().joinpath((in_root_path))
    assert input_base_path.exists() == True, in_root_path+" doesn't exist"

    output_base_path = pathlib.Path.home().joinpath((out_root_path + out_character_name))
    if output_base_path.exists() == False:
        output_base_path.mkdir()
    
    dataset_list = ["motions"]

    print('')
    for level1 in tqdm(input_base_path.iterdir()):
        if level1.is_dir():
            print('motion dataset : ', level1.name)
            print('')
            output_level1_path = output_base_path.joinpath(level1.name)
            if output_level1_path.exists() == False:
                output_level1_path.mkdir()
            if level1.name not in dataset_list:
                print('pass')
                continue
            if level1.is_dir():
                print('Processing sub dir', str(level1), '-----------------------')
                motion_path_list = level1.glob('*.npy')
                for i, motion_path in enumerate(motion_path_list):
                    print('Num', i, motion_path)
                    output_level2_path = output_level1_path.joinpath(motion_path.name)
                    output_final_path = output_level2_path.with_suffix('.npy')
                    
                    output = export_one(skel_tree, motion_path, str(output_final_path), dfs_to_bfs_list, joint_ids)
                    if output == False:
                        print('Failed file, ', motion_path, '!!!!!!!!!!!!!!!!!!!!!!!')
        else:
            print(level1, ' is not a folder, skip!!!')


def main():

    ASSET_ROOT_PATH = "./assets"

    mjcf_path = ASSET_ROOT_PATH + "/robots/mjcf/amp_humanoid.xml"
    in_character_name = "amp_humanoid"
    dfs_skel_tree = SkeletonTree.from_mjcf(mjcf_path, search="Depth")
    bfs_skel_tree = SkeletonTree.from_mjcf(mjcf_path, search="Breadth")
    dfs_to_bfs_list = dfs_to_bfs_order(dfs_skel_tree.node_names, bfs_skel_tree.node_names)
    
    if True:
        mjcf_path2 = ASSET_ROOT_PATH + "/robots/mjcf/amp_humanoid2.xml" # add toe bodies on each foot.
        out_character_name = "amp_humanoid2"
        dfs_skel_tree2 = SkeletonTree.from_mjcf(mjcf_path2, search="Depth")
        bfs_skel_tree2 = SkeletonTree.from_mjcf(mjcf_path2, search="Breadth")
        dfs_to_bfs_list = dfs_to_bfs_order(dfs_skel_tree2.node_names, bfs_skel_tree2.node_names)
        body_names_list = list(set(dfs_skel_tree2.node_names) - set(dfs_skel_tree.node_names))
        joint_ids = []
        for i, name in enumerate(dfs_skel_tree2.node_names):
            if name in body_names_list:
                print('The body does not have joint : ', name)
            else:
                joint_ids.append(i)
        dfs_skel_tree = dfs_skel_tree2
        bfs_skel_tree = bfs_skel_tree2
    
    if out_character_name == None:
        out_character_name = in_character_name

    print('DFS order name : ', dfs_skel_tree.node_names)
    print('BFS order name : ', bfs_skel_tree.node_names)
    print('Mapping order : ', dfs_to_bfs_list)


    '''
    # Mode 1, convert one motion
    #start_frame_idx = 0
    #end_frame_idx = 30 # sliced shape [start_frame_idx, enf_frame_idx]
    in_file_path = ASSET_ROOT_PATH + "/motions/" + file_name
    if start_frame_idx is not None and end_frame_idx is not None:
        out_file_path = ASSET_ROOT_PATH + "/motions/" + out_character_name + file_name[:-4] + "_S" + str(start_frame_idx) + "_E" + str(end_frame_idx) + ".npy"
        convert_one(bfs_skel_tree, in_file_path, out_file_path, dfs_to_bfs_list, joint_ids, (start_frame_idx, end_frame_idx))
    else:
        out_file_path = ASSET_ROOT_PATH + "/motions/" + out_character_name + file_name[:-4] + ".npy"
        convert_one(bfs_skel_tree, in_file_path, out_file_path, dfs_to_bfs_list, joint_ids)
    '''

    # Mode 2, convert all files in the folder
    in_root_path = "/home/asaid/Dev/others/ASE/ase/data/motions/"
    in_root_path = "/home/asaid/Dev/sample_isaac/assets/amp/"
    out_root_path = "/home/asaid/Dev/IsaacLab_AMP_rl-games/assets/motions/"
    convert_all(bfs_skel_tree, out_character_name, in_root_path, out_root_path, dfs_to_bfs_list, joint_ids)
    
    print('Convert Finished')

if __name__ == "__main__":
    main()

