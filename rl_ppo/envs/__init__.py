"""
Importing the folder containing this __init__.py triggers this code.

This code registers the PandaEnv class supporting Gymnasium interface with the
Gymnasium so the agents (training agent, evaluation agent, etc...) can use the
PandaEnv for using the Panda via Gymnasium interface.

Note that the entry point also defines the python file containing the PandaEnv class.
In this case, the file is envs/reach_panda.py.
"""

from gymnasium.envs.registration import register

register(
    id="PandaGymnasium",
    entry_point="envs.reach_panda:PandaEnv",
)
