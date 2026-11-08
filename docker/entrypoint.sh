#!/bin/sh
# Seed /config on first run, refresh the webapp code on every start (without
# touching the generated playlist.json), then hand off to the command.
set -e

mkdir -p /config/references /config/models /config/webapp \
         /config/torch /config/hf /config/.cache
cd /config

# numpy / torch CPU work (training, scoring, PCA) takes about half the cores,
# so a burst never starves Stash or the pages you're on (override to taste)
if [ -z "$OMP_NUM_THREADS" ]; then
  _half=$(( $(nproc 2>/dev/null || echo 2) / 2 )); [ "$_half" -lt 1 ] && _half=1
  export OMP_NUM_THREADS=$_half OPENBLAS_NUM_THREADS=$_half MKL_NUM_THREADS=$_half
fi

# seed config.toml on first run, and refresh it when the bundled defaults change
# (backs up the old file, keeps your [stash] connection block)
python3 /opt/peaks/refresh_config.py || echo "config: refresh skipped"

cp -f /opt/peaks/webapp/index.html \
      /opt/peaks/webapp/megaboard.css \
      /opt/peaks/webapp/megaboard.js \
      /opt/peaks/webapp/peaks.svg \
      /config/webapp/

exec "$@"
