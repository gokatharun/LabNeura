#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Python builds use the authoritative repository src/ and include/ trees via
# CMake. Only non-code packaging assets are mirrored here.
mkdir -p "${ROOT_DIR}/python/assets"

rsync -a "${ROOT_DIR}/assets/cells.lib" "${ROOT_DIR}/python/assets/cells.lib"
rsync -a "${ROOT_DIR}/assets/ice40_hx.lib" "${ROOT_DIR}/python/assets/ice40_hx.lib"
rsync -a "${ROOT_DIR}/assets/sky130_fd_sc_hd__tt_025C_1v80.lib" \
  "${ROOT_DIR}/python/assets/sky130_fd_sc_hd__tt_025C_1v80.lib"
rsync -a "${ROOT_DIR}/assets/SKY130-LICENSE" \
  "${ROOT_DIR}/python/assets/SKY130-LICENSE"
rsync -a "${ROOT_DIR}/assets/SKY130-LIBRARY-NOTICE.md" \
  "${ROOT_DIR}/python/assets/SKY130-LIBRARY-NOTICE.md"

# Also sync README/LICENSE to python root so MANIFEST paths are valid
rsync -a "${ROOT_DIR}/README.md" "${ROOT_DIR}/python/README.md"
rsync -a "${ROOT_DIR}/LICENSE" "${ROOT_DIR}/python/LICENSE"

echo "Synced packaging assets, README.md, and LICENSE into python/"
