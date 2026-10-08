"""Capture complete functional test and scoped lint results from actual execution."""

import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path("experiments/coding_completion_v2")
root.mkdir(parents=True, exist_ok=True)
result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "--junitxml=work/completion_tests.xml"],
    capture_output=True,
    text=True,
    check=False,
)
Path("work/completion_tests.log").write_text(result.stdout + result.stderr)
print(result.stdout[-1200:])
if result.returncode:
    raise SystemExit(result.returncode)
xml = ET.parse("work/completion_tests.xml").getroot()
suites = list(xml) if xml.tag == "testsuites" else [xml]
record = {
    key: sum(int(s.get(key, 0)) for s in suites)
    for key in ("tests", "failures", "errors", "skipped")
}
passed = re.search(r"(\d+) passed", result.stdout)
subtests = re.search(r"(\d+) subtests passed", result.stdout)
record.update(
    pytest_passed_tests=int(passed.group(1)),
    subtests_passed=int(subtests.group(1)) if subtests else 0,
    seconds=sum(float(s.get("time", 0)) for s in suites),
)
(root / "test_results.json").write_text(json.dumps(record, indent=2))
paths = (
    ["src/mindscape/coding", "src/mindscape/sql", "scripts/coding"]
    + [str(p) for p in Path("tests").glob("test_coding*.py")]
    + ["tests/test_sql_vertical.py", "tests/test_lora_checkpoint.py"]
)
subprocess.run([sys.executable, "-m", "ruff", "check", *paths], check=True)
