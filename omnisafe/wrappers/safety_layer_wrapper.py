# Copyright 2022 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Safety layer wrapper."""

import torch
import torch.nn as nn

from omnisafe.wrappers.off_policy_wrapper import OffPolicyEnvWrapper
from omnisafe.wrappers.wrapper_registry import WRAPPER_REGISTRY
from omnisafe.utils.model_utils import build_mlp_network


class LinearCostModel(nn.Module):
    """LinearCostModel."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list, activation: str) -> None:
        super().__init__()
        self.net = build_mlp_network([obs_dim] + hidden_sizes + [act_dim], activation=activation)

    def forward(
        self, obs: torch.Tensor, act: torch.Tensor, previous_cost: torch.Tensor
    ) -> torch.Tensor:
        """forward.

        Args:
            obs: current observation.
            act: current action.
            cost: previous cost.

        Returns:
            the predicted cost of the given observation and action, which is a linear function of
            the action: `pred_cost = previous_cost + gs(obs) * act`.
        """
        gs = self.net(obs)
        pred_cost = previous_cost + torch.matmul(gs.unsqueeze(-2), act.unsqueeze(-1)).squeeze()
        return pred_cost, gs


@WRAPPER_REGISTRY.register
class SafetyLayerWrapper(OffPolicyEnvWrapper):
    """SafetyLayerWrapper."""

    def __init__(
        self,
        env_id,
        max_ep_len,
        cfgs,
        render_mode=None,
    ):
        """Initialize SafetyLayerWrapper.

        Args:
            env_id (str): environment id.
            max_ep_len (int): maximum episode length.
            cfgs (NamedTuple): configuration dictionary.
            render_mode (str): render mode.
        """
        super().__init__(env_id, use_cost=True, max_ep_len=max_ep_len, render_mode=render_mode)

        self.cfgs = cfgs
        # initialize cost model and related optimizer
        self.cost_model = LinearCostModel(
            obs_dim=self.observation_space.shape[0],
            act_dim=self.action_space.shape[0],
            hidden_sizes=cfgs.safety_layer_cfgs.cost_model.hidden_sizes,
            activation=cfgs.safety_layer_cfgs.cost_model.activation,
        )
        self.cost_model_optimizer = torch.optim.Adam(self.cost_model.parameters(), lr=cfgs.model_lr)

        self.cost_limit = cfgs.safety_layer_cfgs.cost_limit

        self.previous_cost = 0
        self.current_observation = None

    def update_cost_model(self, batch: dict) -> None:
        """Update cost model.

        Args:
            batch (dict): batch of data.
        """
        obs, act, cost, previous_cost = (
            batch['obs'],
            batch['act'],
            batch['cost'],
            batch['previous_cost'],
        )
        pred_cost, _ = self.cost_model(obs, act, previous_cost)
        cost_model_loss = torch.mean((pred_cost - cost) ** 2)
        self.cost_model_optimizer.zero_grad()
        cost_model_loss.backward()
        self.cost_model_optimizer.step()
        self.logger.store(**{'Loss/CostModel': cost_model_loss.item()})

    def get_safe_action(self, obs, act, previous_cost):
        """
        Revise the current action to a safe action for the current observation, and cost.

        Args:
            obs (torch.Tensor): current observation.
            act (torch.Tensor): current action.
            previous_cost (torch.Tensor): previous cost.
        """
        pred_cost, gs = self.cost_model(obs, act, previous_cost)
        multiplier = (pred_cost - self.cost_limit) / (
            torch.matmul(gs.unsqueeze(-2), gs.unsqueeze(-1)).squeeze + 1e-8
        )
        multiplier = torch.clamp(multiplier, min=0)
        safe_act = act - multiplier * gs
        return safe_act

    def pretrain_cost_model(self, buf, logger, num_steps=10000, update_interval=100, update_iters=40):
        """Pretrain the cost model using random actions."""
        self.cost_model.train()
        for _ in range(0, num_steps, update_interval):
            self.roll_out(
                actor_critic=None,
                buf=buf,
                logger=logger,
                deterministic=False,
                use_rand_action=True,
            )

            for _ in range(update_iters):
                batch = buf.sample_batch()
                if batch is None:
                    break
                self.update_cost_model(batch)

    def reset(self, seed=None):
        """reset environment."""
        self.previous_cost = 0
        self.current_observation, info = super().reset(seed=seed)
        return self.current_observation, info

    def step(self, action):
        """engine step."""
        safe_action = self.get_safe_action(
            obs=self.current_observation,
            act=action,
            previous_cost=self.previous_cost,
        )
        (
            self.current_observation,
            reward,
            self.previous_cost,
            terminated,
            truncated,
            info,
        ) = super().step(safe_action)
        return self.current_observation, reward, self.previous_cost, terminated, truncated, info

    # pylint: disable=too-many-arguments, too-many-locals
    def roll_out(
        self,
        actor_critic,
        buf,
        logger,
        deterministic,
        use_rand_action,
        ep_steps,
    ):
        """collect data and store to experience buffer. Compared to the original roll_out function,
        this function also collects the previous cost and store it to the buffer."""
        for _ in range(ep_steps):
            ep_ret = self.ep_ret
            ep_len = self.ep_len
            ep_cost = self.ep_cost
            obs = self.curr_o

            if use_rand_action:
                # Get random action
                action = self.env.action_space.sample()
            else:
                # Get action from actor_critic
                action, value, cost_value, _ = actor_critic.step(
                    torch.as_tensor(obs, dtype=torch.float32), deterministic=deterministic
                )
                # Store values for statistic purpose
                if self.use_cost:
                    logger.store(**{'Values/V': value, 'Values/C': cost_value})
                else:
                    logger.store(**{'Values/V': value})

            # Step the env
            # pylint: disable=unused-variable
            prev_cost = self.previous_cost
            obs_next, reward, cost, done, truncated, info = self.step(action)
            ep_ret += reward
            ep_cost += cost
            ep_len += 1
            self.ep_len = ep_len
            self.ep_ret = ep_ret
            self.ep_cost = ep_cost
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            self.curr_o = obs_next
            if not deterministic:
                done = False if ep_len >= self.max_ep_len else done
                buf.store(obs, action, reward, cost, obs_next, done, prev_cost=prev_cost)
                if done or ep_len >= self.max_ep_len:
                    logger.store(
                        **{
                            'Metrics/EpRet': ep_ret,
                            'Metrics/EpLen': ep_len,
                            'Metrics/EpCost': ep_cost,
                        }
                    )
                    self.curr_o, _ = self.env.reset(seed=self.seed)
                    self.ep_ret, self.ep_cost, self.ep_len = 0, 0, 0

            else:
                if done or ep_len >= self.max_ep_len:
                    logger.store(
                        **{
                            'Test/EpRet': ep_ret,
                            'Test/EpLen': ep_len,
                            'Test/EpCost': ep_cost,
                        }
                    )
                    self.curr_o, _ = self.env.reset(seed=self.seed)
                    self.ep_ret, self.ep_cost, self.ep_len = 0, 0, 0

                    return
