# IsaacLab AMP rl-games

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5.0-blue.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.0-green.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository implements [AMP](https://xbpeng.github.io/projects/AMP/index.html) (Adversarial Motion Prior) training environment with **rl_games** algorithm for the **Isaac Lab** framework. \
It allows for quick migration of AMP-based character animation studies implemented with IsaacGym + rl_games to Isaac Lab.

<div align="center">
<img src="media/demo.gif" width="600"/>
</div>

## Quick Start

### Prerequisites

1. **Install Isaac Lab**:
   Follow the [IsaacLab official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html) and install rl_games (tested with Isaac Sim 4.5.0/ IsaacLab 2.0)

2. **Set up Isaac Lab alias**:
   ```bash
   alias ISAACLAB_SH_PATH="~/IsaacLab/isaaclab.sh"
   ```

3. **Clone this repository**:
   ```bash
   git clone https://github.com/Gudegi/IsaacLab_AMP_rl-games.git
   cd IsaacLab_AMP_rl-games
   ```

### Motion Visualization

Preview reference motions interactively:
```bash
ISAACLAB_SH_PATH -p ./scripts/motion_viewer.py
```

### Training

```bash
# Basic training 
ISAACLAB_SH_PATH -p ./scripts/train.py --task=AmpHumanoid-AMP-Direct-v1 --headless

# Resume from checkpoint
ISAACLAB_SH_PATH -p ./scripts/train.py --task=AmpHumanoid-AMP-Direct-v1 --checkpoint=./logs/rl_games/amp_direct/YYYY-MM-DD_HH-MM-SS/nn/model.pth --headless
```

### Inference

```bash
# Play with pre-trained model
ISAACLAB_SH_PATH -p ./scripts/play.py --task=AmpHumanoid-AMP-Direct-v1 --num_envs=1 --checkpoint=./assets/pretrained/run.pth

# Play with your trained model
ISAACLAB_SH_PATH -p ./scripts/play.py --task=AmpHumanoid-AMP-Direct-v1 --num_envs=16 --checkpoint=./logs/rl_games/amp_direct/latest/nn/model.pth

# Record video during inference
ISAACLAB_SH_PATH -p ./scripts/play.py --task=AmpHumanoid-AMP-Direct-v1 --num_envs=1 --checkpoint=./assets/pretrained/run.pth --video
```

## Project Structure

```
IsaacLab_AMP_rl-games/
├── 📁 assets/                          # Assets and data
│   ├── motions/                        # Reference motion files (.npy)
│   │   ├── amp_humanoid_walk.npy
│   │   ├── amp_humanoid_jog.npy
│   │   └── amp_humanoid_run.npy
│   ├── 🤖 robots/                      # Robot descriptions (USD, MJCF)
│   └── 🎯 pretrained/                  # Pre-trained model checkpoints
├── 📁 scripts/                         # Executable scripts
│   ├── train.py                        # Training script
│   ├── play.py                         # Inference script  
│   └── motion_viewer.py                # Motion visualization
├── 📁 source/amp_rlg/                  # Core implementation
│   ├── 🧠 learning/                    # RL-games algorithms
│   │   ├── amp/                        # AMP network
│   │   └── ppo/                        # PPO network
│   ├── 🎮 tasks/                       # Environment definitions
│   │   └── amp/                        # AMP environment
│   └── 🛠️ utils/                        # Utilities and helpers
├── 📁 logs/                            # Training logs and checkpoints
└── 📁 outputs/                         # Hydra configuration outputs
```

## VSCode Settings

1. Run VSCode Tasks by pressing `Ctrl+Shift+P`, selecting **Tasks: Run Task** and running the `setup_python_env` in the drop down menu.

2. If it works well, `launch.json` and `settings.json` are generated. Add the following code to the bottom of `settings.json`:

```json
"${workspaceFolder}/[Relative path to IsaacLab source from this repo]/source/isaaclab_mimic",
"${workspaceFolder}/[Relative path to IsaacLab source from this repo]/source/isaaclab_rl",
"${workspaceFolder}/[Relative path to IsaacLab source from this repo]/source/isaaclab_assets",
"${workspaceFolder}/[Relative path to IsaacLab source from this repo]/source/isaaclab_tasks",
"${workspaceFolder}/[Relative path to IsaacLab source from this repo]/source/isaaclab"
```

## References
This repository borrows code from the following projects:

- [ASE](https://github.com/nv-tlabs/ASE)
- [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs)
- [IsaacLab](https://github.com/isaac-sim/IsaacLab)
- [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate)