"""Classifier Shield plugin for SHIELD."""

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from omnisafe.models.custom_modes.value_model import MLPValueModel
from omnisafe.models.custom_modes.state_classifier_model import MLPStateClassifierModel


class ClassifierShield:
    """
    The classifier shield plugin employs a classifier model to forecast the likelihood
    of an action being safe. It then replaces any unsafe actions with safe ones by
    repeatedly sampling from the action space.
    """
    def __init__(self, env, shield_cfg):
        self._env = env
        self._shield_cfg = shield_cfg
        # *log configuration
        # common parameters
        self._device = torch.device(
            shield_cfg["device"] if torch.cuda.is_available() else "cpu"
        )
        self._dtype = shield_cfg["dtype"]
        self._risk_threshold = shield_cfg["risk_threshold"]
        self._max_resample_times = shield_cfg["max_resample_times"]

        # training parameters
        self._batch_size = shield_cfg["batch_size"]
        self._risk_discount = shield_cfg["risk_discount"]

        # *construct needed components
        risk_model_cfg = shield_cfg["risk_model"]
        self._risk_model = MLPValueModel(
            obs_dim=env.observation_space.shape[0],
            hidden_sizes=risk_model_cfg["hidden_sizes"],
            activation=risk_model_cfg["activation"],
            output_activation=risk_model_cfg["output_activation"],
            weight_initialization_mode=risk_model_cfg["weight_initialization_mode"],
            dtype=self._dtype,
        ).to(self._device)
        self._risk_optimizer = torch.optim.Adam(self._risk_model.parameters())

        classifier_model_cfg = shield_cfg["classifier_model"]
        self._classifier_model = MLPStateClassifierModel(
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            num_classes=2,
            hidden_sizes=classifier_model_cfg["hidden_sizes"],
            activation=classifier_model_cfg["activation"],
            output_activation=classifier_model_cfg["output_activation"],
            weight_initialization_mode=classifier_model_cfg[
                "weight_initialization_mode"
            ],
            dtype=self._dtype,
        ).to(self._device)
        self._classifier_optimizer = torch.optim.Adam(
            self._classifier_model.parameters()
        )

    def _update_risk_model(self, batch: dict):
        """Update the risk model."""
        dataset = TensorDataset(
            batch["obs"],
            batch["act"],
            batch["next_obs"],
            batch["risk"],
            batch["terminated"],
            # batch["truncated"],
        )
        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
        loss_list = []
        bar = tqdm(dataloader)
        for obs, act, next_obs, risk, terminated in bar:
            # for obs, act, next_obs, risk, terminated, truncated in dataloader:
            # compute loss
            target = (
                risk
                + (1 - risk) * self._risk_discount * self._risk_model(next_obs).detach()
            )
            target = torch.where(terminated, risk, target)

            loss = torch.nn.MSELoss()(self._risk_model(obs), target)
            loss_list.append(loss.item())
            bar.set_description(
                f"Training risk model - risk loss: {np.mean(loss_list):.4f}"
            )

            # update model
            self._risk_optimizer.zero_grad()
            loss.backward()
            self._risk_optimizer.step()

        return np.mean(loss_list)

    def _update_classifier_model(self, batch: dict):
        """Update the classifier model."""
        dataset = TensorDataset(
            batch["obs"],
            batch["act"],
            batch["next_obs"],
        )
        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
        loss_list = []

        bar = tqdm(dataloader)
        for obs, act, next_obs in bar:
            # for obs, act, next_obs in dataloader:
            # label safe actions as 0 and unsafe actions as 1
            risk = self._risk_model(next_obs)
            label = torch.where(
                risk > self._risk_threshold,
                torch.ones_like(risk),
                torch.zeros_like(risk),
            )

            # compute loss
            loss = torch.nn.CrossEntropyLoss()(
                self._classifier_model(obs, act), label.long()
            )
            loss_list.append(loss.item())
            bar.set_description(
                f"Training classifier model - classifier loss: {np.mean(loss_list):.4f}"
            )

            # update model
            self._classifier_optimizer.zero_grad()
            loss.backward()
            self._classifier_optimizer.step()

        return np.mean(loss_list)

    def update(self, batch: dict, warmup: bool = False):
        """Update the shield plugin."""
        risk_loss = self._update_risk_model(batch)
        if warmup:
            return {"risk_loss": risk_loss}

        classifier_loss = self._update_classifier_model(batch)
        return {"risk_loss": risk_loss, "classifier_loss": classifier_loss}

    def predict(self, obs: torch.Tensor, act: torch.Tensor):
        """Predict the risk of an action."""

        if obs.dim() == 1 and act.dim() == 1:
            obs = obs.unsqueeze(0)
            act = act.unsqueeze(0)
        elif obs.dim() != act.dim():
            raise ValueError(
                "The dimension of obs and act should be the same, "
                f"but got {obs.dim()} and {act.dim()}."
            )

        with torch.no_grad():
            return self._classifier_model(obs, act)

    def save_model(self, path: str):
        """Save the shield plugin."""
        torch.save(
            {
                "risk_model": self._risk_model.state_dict(),
                "risk_optimizer": self._risk_optimizer.state_dict(),
                "classifier_model": self._classifier_model.state_dict(),
                "classifier_optimizer": self._classifier_optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path: str):
        """Load the shield plugin."""
        checkpoint = torch.load(path)
        self._risk_model.load_state_dict(checkpoint["risk_model"])
        self._risk_optimizer.load_state_dict(checkpoint["risk_optimizer"])
        self._classifier_model.load_state_dict(checkpoint["classifier_model"])
        self._classifier_optimizer.load_state_dict(checkpoint["classifier_optimizer"])
