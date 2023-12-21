#!/usr/bin/env bash
set -euo pipefail

PARALLELISM="$(getconf _NPROCESSORS_ONLN)"

convert_dot() {
  find viz -name "*.dot" | xargs -P "$PARALLELISM" -I '{}' \
    sh -c 'dot -Tpng "$1" -o "${1%.dot}.png"' _ '{}'
}

make_video() {
  ffmpeg -framerate 8 -i viz/%06d.png \
    -vf "scale=ceil(iw/2)*2:ceil(ih/2)*2" \
    -c:v libx265 -crf 34 \
    -pix_fmt yuv420p viz.mp4
}

set -x

convert_dot
make_video
