# isaaclab-project-template

A clean, batteries-included starting point for [IsaacLab](https://github.com/isaac-sim/IsaacLab)
locomotion projects.  Clone or use it as a **GitHub template**, swap in your
own robot, and start training.

---

## Quick Start

### 1. Use this template

Click **"Use this template"** on GitHub, or clone directly:

```bash
git clone https://github.com/<you>/my-isaaclab-project
cd my-isaaclab-project
```

### 2. Configure VS Code

Edit `.vscode/settings.json` and replace `PATH_TO_ISAACLAB` with your local IsaacLab installation directory.

### 3. Plug in your robot
Copy your `robot.usd` file to `asset`
Open `assets/robot_cfg.py` and replace `ROBOT_CONFIG` with your own
`ArticulationCfg`.  Examples from `isaaclab_assets`:

```python
# Quadruped
from isaaclab_assets import UNITREE_GO2_CFG
ROBOT_CONFIG = UNITREE_GO2_CFG

# Biped
from isaaclab_assets import H1_CFG
ROBOT_CONFIG = H1_CFG
```

### 4. Adapt the task configs

Search for `# TODO` in `tasks/template_flat.py` and `tasks/template_terrain.py`.
The three most important spots are:

| Location | What to change |
|---|---|
| `MySceneCfg.robot` | `init_state.pos` z-offset so the robot spawns above ground |
| `ActionsCfg.joint_pos` | `joint_names` regex to match your robot's joints |
| `RewardsCfg.feet_air_time` / `TerminationsCfg.base_contact` | `body_names` regex to match your feet / base link |

---

## Project Structure

```
├── assets/
│   ├── robot_cfg.py        ← ROBOT_CONFIG placeholder  ← edit this first
│   ├── random_action.py    ← standalone spawn + random-action demo
│   └── spawn_robot.py      ← standalone interactive scene demo
│
├── tasks/
│   ├── __init__.py         ← Gym task registration
│   ├── template_flat.py    ← flat-terrain env  (MyFlatEnvCfg)
│   ├── template_terrain.py ← rough-terrain env (MyTerrainEnvCfg)
│   ├── agents/
│   │   └── rsl_rl_ppo.py   ← PPO runner hyper-parameters
│   └── mdp/
│       ├── dummy_action.py ← example custom action term (extend this)
│       ├── observations.py ← custom observation functions
│       ├── terminations.py ← joint-limit termination helper
│       └── terrain_cfg.py  ← MY_ROUGH_TERRAINS_CFG definition
│
└── scripts/
    ├── zero_agent.py       ← sanity-check: zero-action rollout
    ├── random_agent.py     ← sanity-check: random-action rollout
    └── rsl_rl/
        ├── train.py        ← single-GPU / multi-GPU training entry-point
        └── play.py         ← policy playback + JIT/ONNX export
```

---

## Training

**Single GPU**

```bash
isaaclab scripts/rsl_rl/train.py --task template_flat_v0 --num_envs 4096
```

**Multi GPU**

```bash
isaaclab -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
    scripts/rsl_rl/train.py --task template_flat_v0 --headless --distributed
```

**Rough terrain**

```bash
isaaclab scripts/rsl_rl/train.py --task template_terrain_v0 --num_envs 4096
```

## Playback

```bash
isaaclab scripts/rsl_rl/play.py --task template_flat_v0 --num_envs 16
```

## Fine-tune from a checkpoint

```bash
# Symlink a previous run's log directory
ln -s PATH_TO_TRAINED_LOG PATH_TO_FINETUNE_LOG

isaaclab -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
    scripts/rsl_rl/train.py --task template_flat_v0 --headless --distributed \
    --resume \
    --load_run 2026-01-28_16-57-50_init \
    --checkpoint model_14950.pt
```

---

## Registered Gym Tasks

| Task ID | Env class | Description |
|---|---|---|
| `template_flat_v0` | `MyFlatEnvCfg` | Flat ground, velocity tracking |
| `template_terrain_v0` | `MyTerrainEnvCfg` | Rough terrain + height scan + curriculum |

---

## Adding a New Task

1. Copy `tasks/template_flat.py` → `tasks/my_task.py`
2. Rename the top-level `@configclass` to e.g. `MyTaskEnvCfg`
3. Register it in `tasks/__init__.py`:

```python
gym.register(
    id="my_task_v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_task:MyTaskEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo:RSLRLPPORunnerCfg",
    },
)
```
