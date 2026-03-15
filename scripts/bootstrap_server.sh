#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/autodl-tmp/qwen3.5-nano}"
VENV_DIR="${PROJECT_ROOT}/.venv"
export PIP_CACHE_DIR="${HOME}/autodl-tmp/pip-cache"
export UV_CACHE_DIR="${HOME}/autodl-tmp/uv-cache"

mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

mkdir -p "$HOME/.config/pip"
cat > "$HOME/.config/pip/pip.conf" <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
timeout = 120
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

cat > "$HOME/.bash_aliases" <<'EOF'
alias ll='ls -alF'
alias gs='git status -sb'
alias py='python3'
EOF

cat > "$HOME/.bashrc_qwen35_nano_env" <<'EOF'
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$HOME/autodl-tmp/hf
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=$HOME/autodl-tmp/pip-cache
export UV_CACHE_DIR=$HOME/autodl-tmp/uv-cache
export OMP_NUM_THREADS=1
EOF

if ! grep -q "bashrc_qwen35_nano_env" "$HOME/.bashrc"; then
  printf '\n[ -f "$HOME/.bashrc_qwen35_nano_env" ] && source "$HOME/.bashrc_qwen35_nano_env"\n' >> "$HOME/.bashrc"
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends git-lfs tmux rsync python3-venv build-essential
git lfs install

mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

if [ ! -d .venv ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
source "$HOME/.bashrc_qwen35_nano_env"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

echo "Bootstrap complete."
echo "Project root: $PROJECT_ROOT"
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Install torch after the GPU is attached: bash scripts/install_torch.sh cu124"
