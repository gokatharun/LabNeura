#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Ensure target directories exist inside python package
mkdir -p "${ROOT_DIR}/python/src/labneura" "${ROOT_DIR}/python/include/labneura" \
  "${ROOT_DIR}/python/assets"

# Sync C++ sources and headers into python/ tree so packaging works in isolated build dirs
rsync -a --delete "${ROOT_DIR}/src/labneura/" "${ROOT_DIR}/python/src/labneura/"
rsync -a --delete "${ROOT_DIR}/include/labneura/" "${ROOT_DIR}/python/include/labneura/"
rsync -a "${ROOT_DIR}/assets/cells.lib" "${ROOT_DIR}/python/assets/cells.lib"

# Also sync README/LICENSE to python root so MANIFEST paths are valid
rsync -a "${ROOT_DIR}/README.md" "${ROOT_DIR}/python/README.md"
rsync -a "${ROOT_DIR}/LICENSE" "${ROOT_DIR}/python/LICENSE"

echo "Synced src/, include/, README.md, LICENSE into python/ for packaging"