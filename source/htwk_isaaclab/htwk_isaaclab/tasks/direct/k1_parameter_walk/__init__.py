"""K1 parameter-walk direct RL task."""

import gymnasium as gym

from . import agents


gym.register(
    id="HTWK-K1-ParameterWalk-Direct-v0",
    entry_point=f"{__name__}.k1_parameter_walk_env:K1ParameterWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.k1_parameter_walk_env:K1ParameterWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1ParameterWalkPPORunnerCfg",
    },
)

gym.register(
    id="HTWK-K1-ParameterWalk-Direct-Play-v0",
    entry_point=f"{__name__}.k1_parameter_walk_env:K1ParameterWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.k1_parameter_walk_env:K1ParameterWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1ParameterWalkPPORunnerCfg",
    },
)
