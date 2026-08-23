#!/bin/sh
# Seed /config on first run, refresh the webapp code on every start (without
# touching the generated playlist.json), then hand off to the command.
set -e

mkdir -p /config/references /config/models /config/webapp
cd /config

# Seed the model cache from the image's pre-baked copy when /config is empty or
# not persisted, so the ~5GB of weights is never re-downloaded. -n never
# overwrites an existing / user-selected variant; this is a local copy, not a
# network fetch (skips instantly once /config already holds the weights).
if [ -d /opt/peaks/model-preload ]; then
  mkdir -p /config/torch /config/hf
  cp -rn /opt/peaks/model-preload/torch/. /config/torch/ 2>/dev/null || true
  cp -rn /opt/peaks/model-preload/hf/. /config/hf/ 2>/dev/null || true
fi

# seed config.toml on first run, and refresh it when the bundled defaults change
# (backs up the old file, keeps your [stash] connection block)
python3 /opt/peaks/refresh_config.py || echo "config: refresh skipped"

cp -f /opt/peaks/webapp/index.html \
      /opt/peaks/webapp/megaboard.css \
      /opt/peaks/webapp/megaboard.js \
      /config/webapp/

exec "$@"
