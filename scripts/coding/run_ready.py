"""Local finite job dependency runner; no scheduler, recurrence, or remote calls."""

import argparse
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--wait-for", action="append", default=[])
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
while any(not Path(path).exists() for path in a.wait_for):
    time.sleep(5)
command = a.command[1:] if a.command and a.command[0] == "--" else a.command
if not command:
    raise RuntimeError("Command required")
raise SystemExit(subprocess.run(command, check=False).returncode)
