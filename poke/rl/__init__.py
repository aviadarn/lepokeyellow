"""Reinforcement learning. Everything under this package is TRAINING TIME.

Training-time code may read RAM, restore save states and compute rewards. The
play-time surface is `poke.contract.PlayTimeEnv`, and the only module here
that a policy touches through it is `poke.rl.policy`, which sees embeddings.
"""
