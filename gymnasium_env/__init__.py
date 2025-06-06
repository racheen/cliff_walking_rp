from gymnasium.envs.registration import register

register(
    id="gymnasium_env/CliffWalker",
    entry_point="gymnasium_env.envs:CliffWalker",
)

register(
    id="gymnasium_env/CliffWalkerPositive",
    entry_point="gymnasium_env.envs:CliffWalkerPositive",
)
