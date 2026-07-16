#!/bin/sh
# Seed /config on first run, refresh the webapp code on every start (without
# touching the generated playlist.json), then hand off to the command.
set -e

mkdir -p /config/references /config/models /config/webapp
cd /config

# seed config.toml on first run, and refresh it when the bundled defaults change
# (backs up the old file, keeps your [stash] connection block)
python3 /opt/peaks/refresh_config.py || echo "config: refresh skipped"

cp -f /opt/peaks/webapp/index.html \
      /opt/peaks/webapp/megaboard.css \
      /opt/peaks/webapp/megaboard.js \
      /config/webapp/

exec "$@"
