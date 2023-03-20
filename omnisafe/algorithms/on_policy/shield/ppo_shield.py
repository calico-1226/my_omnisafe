"""Implementation of PPO algorithm with shield."""

import time
from typing import Any, Dict, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter import ShieldAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.utils import distributed

@registry.register
class PPOShield(PPO):
    """The Proximal Policy Optimization (PPO) algorithm with shield."""

    def _init_env(self) -> None:
        self._env = ShieldAdapter(
            self._env_id, self._cfgs.train_cfgs.vector_env_nums, self._seed, self._cfgs
        )
        assert (self._cfgs.algo_cfgs.update_cycle) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, ('The number of steps per epoch is not divisible by the number of ' 'environments.')
        self._steps_per_epoch = (
            self._cfgs.algo_cfgs.update_cycle
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )
