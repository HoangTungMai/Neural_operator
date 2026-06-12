#!/usr/bin/env bash
# Giai doan B — cai Isaac Sim 4.5 (khong TacEx) de sinh ground-truth PhysX FEM.
#
# Yeu cau da xac minh tren may nay:
#   GPU RTX 2000 Ada 16GB, driver 580/CUDA13 (OK), Ubuntu 22.04, GLIBC 2.35 (OK >=2.34)
#   RAM 15GB  -> DUOI khuyen nghi 32GB: chay headless/single-env, scene nho.
#
# Chay tung buoc (KHONG chay het mot lan dau tien) — co buoc can EULA va tai nang.
set -e

ENV_NAME="${ENV_NAME:-isaaclab}"
MINICONDA_DIR="$HOME/miniconda3"

step() { echo; echo "=== [$1] $2 ==="; }

# --- 1. Miniconda (neu chua co) ---
step 1 "Miniconda"
if ! command -v conda >/dev/null 2>&1 && [ ! -d "$MINICONDA_DIR" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$MINICONDA_DIR"
fi
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# --- 2. Conda env python 3.10 (Isaac Sim 4.x yeu cau 3.10) ---
step 2 "conda env $ENV_NAME (python 3.10)"
conda create -y -n "$ENV_NAME" python=3.10 || true
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# --- 3. PyTorch CUDA (khop driver) ---
step 3 "PyTorch CUDA"
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# --- 4. Isaac Sim 4.5 (pip) ---
# GLIBC 2.35 OK. EULA: dat bien moi truong de tu dong chap nhan khi chay headless.
step 4 "Isaac Sim 4.5"
export OMNI_KIT_ACCEPT_EULA=YES
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# --- 5. Smoke test (headless) ---
# Lan dau keo extension ~10+ phut. Neu thay viewport/log khoi tao xong la OK.
step 5 "smoke test (headless)"
export OMNI_KIT_ACCEPT_EULA=YES
python - <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.usd
from pxr import UsdGeom, Usd
stage = omni.usd.get_context().get_stage()
print("Isaac Sim stage OK:", stage is not None)
app.close()
print("SMOKE TEST PASSED")
PY

echo
echo "=== Isaac Sim san sang. Buoc tiep: python scripts/isaac_extract_groundtruth.py ==="
echo "(Activate env truoc: conda activate $ENV_NAME)"
