"""Shield Adapter."""
from tqdm import tqdm
from typing import Dict, Optional

import torch
import numpy as np

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer, VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config


class ShieldAdapter(OnPolicyAdapter):
    """Shield Adapter."""

    def __init__(  # pylint: disable=too-many-arguments
        self, env_id: str, num_envs: int, seed: int, cfgs: Config
    ) -> None:
        super().__init__(env_id, num_envs, seed, cfgs)

    def random_exploration(self, steps: int) -> None:
        """Random explore the environment and store the data in the buffer."""
        assert steps % self._env.num_envs == 0, "steps must be divisible by num_envs"

        buf = VectorOffPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=steps // self._env.num_envs,
            batch_size=1,
            num_envs=self._env.num_envs,
        )

        _, info = self.reset()
        original_obs = info["original_obs"]

        bar = tqdm(range(steps // self._env.num_envs))
        # for step in range(steps // self._env.num_envs):
        for step in bar:
            act = [self._env.action_space.sample() for _ in range(self._env.num_envs)]
            act = torch.from_numpy(np.array(act))

            next_obs, reward, cost, terminated, truncated, info = self.step(act)
            next_original_obs = info["original_obs"]
            done = terminated.bool() | truncated.bool()
            done_idx = done.nonzero(as_tuple=True)[0]

            if len(done_idx) > 0:
                final_obs = [
                    torch.from_numpy(obs)
                    for obs in info.get("final_observation", next_original_obs)
                ]
                final_obs = torch.stack(final_obs, dim=0).to(torch.float32)
                next_original_obs[done_idx] = final_obs[done_idx]

            buf.store(
                obs=original_obs,
                act=act,
                reward=info["original_reward"],
                cost=info["original_cost"],
                done=terminated,
                next_obs=next_original_obs,
            )

            original_obs = info["original_obs"]

        print("Collected {} random samples.".format(buf._size * self._env.num_envs))
        return {key: value[: buf._size] for key, value in buf.data.items()}
