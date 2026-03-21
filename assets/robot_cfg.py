# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot asset configuration.

TODO: Replace ROBOT_CONFIG with your own robot's ArticulationCfg.

Common configurations available in ``isaaclab_assets``:
    - Quadruped : ANYMAL_C_CFG, ANYMAL_D_CFG, UNITREE_A1_CFG, UNITREE_GO2_CFG
    - Biped     : H1_CFG, G1_CFG
    - Arm       : FRANKA_PANDA_CFG, UR10_CFG

Quick swap example::

    from isaaclab_assets import UNITREE_GO2_CFG
    ROBOT_CONFIG = UNITREE_GO2_CFG

For a custom robot (USD + URDF workflow):
    1. Place your robot USD under ``assets/`` (e.g. ``assets/my_robot.usd``).
    2. Create an ``ArticulationCfg`` pointing to that USD.
    3. Assign it to ``ROBOT_CONFIG`` below.
"""

# Default placeholder: ANYmal C quadruped (always available with IsaacLab).
# Replace this import and ROBOT_CONFIG with your own robot.
from isaaclab_assets import ANYMAL_C_CFG  # noqa: F401

# TODO: Replace with your robot's ArticulationCfg
ROBOT_CONFIG = ANYMAL_C_CFG
