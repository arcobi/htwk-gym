"""Run Isaac Lab's RSL-RL trainer with HTWK task registrations loaded."""

from __future__ import annotations

import runpy
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_ROOT", REPO_ROOT.parent / "IsaacLab")).expanduser()
RSL_RL_DIR = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"

if not RSL_RL_DIR.exists():
    raise FileNotFoundError(
        f"Could not find Isaac Lab RSL-RL scripts at {RSL_RL_DIR}. "
        "Set ISAACLAB_ROOT to your Isaac Lab checkout."
    )

sys.path.insert(0, str(REPO_ROOT / "source" / "htwk_isaaclab"))
sys.path.insert(0, str(RSL_RL_DIR))

import htwk_isaaclab  # noqa: F401,E402

runpy.run_path(str(RSL_RL_DIR / "train.py"), run_name="__main__")
