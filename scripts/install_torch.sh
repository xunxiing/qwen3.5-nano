#!/usr/bin/env bash
set -euo pipefail

CHANNEL="${1:-cu124}"

case "$CHANNEL" in
  cu124)
    INDEX_URL="https://download.pytorch.org/whl/cu124"
    ;;
  cu121)
    INDEX_URL="https://download.pytorch.org/whl/cu121"
    ;;
  cpu)
    INDEX_URL="https://download.pytorch.org/whl/cpu"
    ;;
  *)
    echo "Unsupported channel: $CHANNEL" >&2
    echo "Use one of: cu124, cu121, cpu" >&2
    exit 1
    ;;
esac

python -m pip install --upgrade torch torchvision --index-url "$INDEX_URL"
