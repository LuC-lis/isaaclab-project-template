# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Example of a custom action term.

This module provides a minimal ``DummyJointPositionAction`` that forwards the
network output directly to joint position targets (with an optional scale and
default-pose offset).  Use it as a starting point for your own action terms,
e.g. a Central Pattern Generator (CPG), impedance controller, or delta-action
wrapper.

Usage in a task config::

    from tasks.mdp.dummy_action import DummyJointPositionActionCfg

    @configclass
    class ActionsCfg:
        joint_pos = DummyJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
        )
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DummyJointPositionAction(ActionTerm):
    """Minimal custom action term: scale → (optional default offset) → joint position targets.

    Extend or replace the logic in ``apply_actions`` / ``process_actions`` to
    implement more complex controllers (e.g. CPG, residual policy, IK).
    """

    cfg: DummyJointPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: DummyJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        # Resolve controlled joints
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    # ------------------------------------------------------------------
    # Properties required by ActionTerm
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor) -> None:
        """Scale actions and optionally add the default joint positions."""
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions * self.cfg.scale

        if self.cfg.use_default_offset:
            default_pos = self._asset.data.default_joint_pos[:, self._joint_ids]
            self._processed_actions += default_pos

    def apply_actions(self) -> None:
        """Send processed actions to the physics simulation."""
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0


@configclass
class DummyJointPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`DummyJointPositionAction`."""

    class_type: type[ActionTerm] = DummyJointPositionAction

    # TODO: Narrow joint_names to the joints you actually want to control.
    joint_names: list[str] = [".*"]
    """Regex patterns selecting the controlled joints."""

    scale: float = 0.5
    """Multiplier applied to the network output before sending to the robot."""

    use_default_offset: bool = True
    """If True, add the robot's default joint positions so the policy outputs
    *deltas* from the rest pose rather than absolute positions."""
