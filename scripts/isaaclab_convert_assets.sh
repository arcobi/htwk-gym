#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/IsaacLab}"

if [[ ! -x "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
  echo "Could not find Isaac Lab at ISAACLAB_ROOT=${ISAACLAB_ROOT}" >&2
  echo "Set ISAACLAB_ROOT to your Isaac Lab checkout, for example:" >&2
  echo "  export ISAACLAB_ROOT=/path/to/IsaacLab" >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/resources/isaaclab"

"${ISAACLAB_ROOT}/isaaclab.sh" -p "${ISAACLAB_ROOT}/scripts/tools/convert_urdf.py" \
  "${REPO_ROOT}/resources/K1/K1_locomotion.urdf" \
  "${REPO_ROOT}/resources/isaaclab/K1_locomotion.usd" \
  --headless \
  --merge-joints \
  --joint-target-type position \
  --joint-stiffness 100.0 \
  --joint-damping 2.0
