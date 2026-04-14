#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  install_deps.sh  –  Fetch third-party header-only libraries
#  Usage:  bash scripts/install_deps.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
THIRD_PARTY="$ROOT_DIR/third-party"

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
RESET="\033[0m"

log()  { echo -e "${CYAN}[deps]${RESET} $*"; }
ok()   { echo -e "${GREEN}[done]${RESET} $*"; }
skip() { echo -e "${YELLOW}[skip]${RESET} $*"; }

mkdir -p "$THIRD_PARTY"

# ── Eigen ─────────────────────────────────────────────────────
EIGEN_DIR="$THIRD_PARTY/eigen"
EIGEN_URL="https://gitlab.com/libeigen/eigen.git"
EIGEN_TAG="3.4.0"

if [ -d "$EIGEN_DIR/.git" ]; then
    skip "Eigen already present at third-party/eigen  (remove it to re-clone)"
else
    log "Cloning Eigen ${EIGEN_TAG} ..."
    git clone --depth 1 --branch "$EIGEN_TAG" "$EIGEN_URL" "$EIGEN_DIR"
    ok "Eigen installed → third-party/eigen"
fi

# ── MiniDNN ───────────────────────────────────────────────────
MINIDNN_DIR="$THIRD_PARTY/MiniDNN"
MINIDNN_URL="https://github.com/yixuan/MiniDNN.git"

if [ -d "$MINIDNN_DIR/.git" ]; then
    skip "MiniDNN already present at third-party/MiniDNN  (remove it to re-clone)"
else
    log "Cloning MiniDNN (latest main) ..."
    git clone --depth 1 "$MINIDNN_URL" "$MINIDNN_DIR"
    ok "MiniDNN installed → third-party/MiniDNN"
fi

# ── nlohmann/json ─────────────────────────────────────────────
JSON_DIR="$THIRD_PARTY/nlohmann"
JSON_URL="https://github.com/nlohmann/json/releases/latest/download/json.hpp"

if [ -f "$JSON_DIR/json.hpp" ]; then
    skip "nlohmann/json already present at third-party/nlohmann/json.hpp"
else
    log "Downloading nlohmann/json (single header) ..."
    mkdir -p "$JSON_DIR"
    curl -sSL "$JSON_URL" -o "$JSON_DIR/json.hpp"
    ok "nlohmann/json installed → third-party/nlohmann/json.hpp"
fi

echo ""
echo -e "${GREEN}All dependencies installed.${RESET}"
echo -e "Now run:  ${CYAN}cmake -B build && cmake --build build${RESET}"
