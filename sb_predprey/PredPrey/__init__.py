import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

if __name__ == "PredPrey":
    register(
        id='PredPreyEnv-v0',
        entry_point='PredPrey.envs:PredPreyEnv',
    )


