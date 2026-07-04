#!/bin/sh
# Seed /config on first run, refresh the webapp code on every start (without
# touching the generated playlist.json), then hand off to the command.
set -e

mkdir -p /config/references /config/models /config/webapp
cd /config

[ -f config.toml ] || cp /opt/peaks/config.example.toml config.toml

cp -f /opt/peaks/webapp/index.html \
      /opt/peaks/webapp/megaboard.css \
      /opt/peaks/webapp/megaboard.js \
      /config/webapp/

exec "$@"
