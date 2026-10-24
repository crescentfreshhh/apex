#!/usr/bin/env python3
"""A stand-in for rclone in the seedbox job tests: `lsf -R --files-only` lists
a local folder, `copyto` copies a file and prints rclone-style JSON stats on
stderr. Remotes are `fake:<local path>`. A file named *FAILME* fails to copy;
FAKE_RCLONE_LSF_FAIL=1 makes listing fail; FAKE_RCLONE_SLEEP slows copies."""

import json
import os
import shutil
import sys
import time


def local(remote: str) -> str:
    return remote.split(":", 1)[1] if remote.startswith("fake:") else remote


def main(argv):
    cmd = argv[0]
    if cmd == "lsf":
        if os.environ.get("FAKE_RCLONE_LSF_FAIL") == "1":
            print("connection refused", file=sys.stderr)
            return 1
        root = local(argv[-1])
        if not os.path.isdir(root):
            return 3
        for d, _, files in os.walk(root):
            for f in files:
                print(os.path.relpath(os.path.join(d, f), root))
        return 0
    if cmd == "copyto":
        src, dst = local(argv[1]), argv[2]
        if "FAILME" in src:
            print(json.dumps({"level": "error", "msg": "copy failed", "time": "t"}), file=sys.stderr)
            return 1
        size = os.path.getsize(src)
        for frac in (0.5, 1.0):
            print(json.dumps({"level": "info", "msg": "stats", "time": "t",
                              "stats": {"bytes": int(size * frac), "totalBytes": size, "speed": 1e6}}),
                  file=sys.stderr, flush=True)
            time.sleep(float(os.environ.get("FAKE_RCLONE_SLEEP", "0")))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        print(json.dumps({"level": "info", "msg": f"Copied (new): {os.path.basename(dst)}", "time": "t"}),
              file=sys.stderr)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
