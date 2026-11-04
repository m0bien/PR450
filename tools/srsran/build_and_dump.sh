#!/usr/bin/env bash
# Build srsRAN 4G's PHY library (if needed) and the LTE vector dumper, then write
# reference vectors to tests/data/srsran/. Run inside WSL / Linux:
#
#   bash tools/srsran/build_and_dump.sh
#
# Requires: build-essential cmake libfftw3-dev libmbedtls-dev libboost-program-options-dev
set -euo pipefail

SRSRAN_DIR="${SRSRAN_DIR:-$HOME/src/srsRAN_4G}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$REPO/tests/data/srsran"
CELL_IDS=(0 1 2 5 101 250 503)

if [ ! -d "$SRSRAN_DIR" ]; then
  git clone --depth 1 https://github.com/srsran/srsRAN_4G.git "$SRSRAN_DIR"
fi
if [ ! -f "$SRSRAN_DIR/build/lib/src/phy/libsrsran_phy.a" ]; then
  mkdir -p "$SRSRAN_DIR/build"
  (cd "$SRSRAN_DIR/build" &&
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DENABLE_UHD=OFF -DENABLE_BLADERF=OFF -DENABLE_SOAPYSDR=OFF -DENABLE_ZEROMQ=OFF \
      -DENABLE_SRSUE=OFF -DENABLE_SRSENB=OFF -DENABLE_SRSEPC=OFF -DENABLE_GUI=OFF >/dev/null &&
    make -j"$(nproc)" srsran_phy)
fi

BIN="$(mktemp -d)/srsran_lte_vectors"
gcc -O2 -std=gnu11 -o "$BIN" "$HERE/srsran_lte_vectors.c" \
  -I"$SRSRAN_DIR/lib/include" -I"$SRSRAN_DIR/build/lib/include" \
  "$SRSRAN_DIR/build/lib/src/phy/libsrsran_phy.a" \
  -lfftw3f -lm -lpthread -lstdc++

mkdir -p "$OUT"
"$BIN" "$OUT" "${CELL_IDS[@]}"
git -C "$SRSRAN_DIR" log -1 --format="srsRAN_4G %H (%cd)" > "$OUT/SOURCE.txt"
echo "done: $(ls "$OUT"/*.bin | wc -l) vector files, $(cat "$OUT/SOURCE.txt")"
