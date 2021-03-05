from gym.envs.registration import register

register(
    id='phypacman-v0',
    entry_point='PacMan.envs:PacManEnv',
)
