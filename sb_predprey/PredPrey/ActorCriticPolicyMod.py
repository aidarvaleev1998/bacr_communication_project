from typing import Optional, List, Union, Dict, Type, Any

import gym
from gym.vector.utils import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
from stable_baselines3.common.torch_layers import MlpExtractor


class ActorCriticPolicyMod(ActorCriticPolicy):
    def __init__(self,
                 *args,
                 **kwargs):
        action_space = spaces.Box(-1., 1., shape=(3,), dtype='float32')
        args = (args[0], action_space, args[2])
        super().__init__(*args, **kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        first_r_obs, second_r_obs = obs.split(obs.shape[1] // 2, dim=1)
        actions_1, values_1, log_prob_1 = super().forward(first_r_obs)
        actions_2, values_2, log_prob_2 = super().forward(second_r_obs)

        conc_actions = th.hstack((actions_1, actions_2))
        # TODO: check if we can approximate the values in such a way
        conc_values = (values_1 + values_2) / 2
        conc_log = (log_prob_1 + log_prob_2) / 2

        return conc_actions, conc_values, conc_log

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim // 2,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
