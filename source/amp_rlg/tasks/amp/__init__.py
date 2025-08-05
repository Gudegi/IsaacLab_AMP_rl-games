
"""
AMP environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="AmpHumanoid-AMP-Direct-v1",
    entry_point=f"{__name__}.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:AmpEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_amp_ppo_cfg.yaml",
    },
)
