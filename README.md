# IsaacLab_AMP_rl-games

This repository contains both **AMP** (Adversarial Motion Prior) training environment and **rl_games** training algorithm for **Isaac Lab** framework.

#### Target Readers: Those who want to quickly migrate from Isaac Gym to Isaac Lab.

<div float="center">
<img src="media/demo.gif" width="600"/>
</div>

## How to use 
1. Install [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html) with rl_games (tested in Isaac Sim 4.5.0/ IsaacLab 2.0)
2. Set alias like `alias ISAACLAB_SH_PATH="~/IsaacLab/isaaclab.sh"`
3. Clone this repository

## Motion Viewer
```Bash
    ISAACLAB_SH_PATH -p ./amp_rlg/motion_viewer.py 
```

## Training
```Bash
    ISAACLAB_SH_PATH -p ./amp_rlg/train.py --task=AmpHumanoid-AMP-Direct-v1 --headless
```

## Inference
```Bash
    ISAACLAB_SH_PATH -p ./amp_rlg/play.py --task=AmpHumanoid-AMP-Direct-v1 --num_envs=1 --checkpoint=./assets/pretrained/run.pth
```

## References
This repository borrows code from the following projects:

IsaacGymEnvs : https://github.com/isaac-sim/IsaacGymEnvs

IsaacLab : https://github.com/isaac-sim/IsaacLab