"""Finite completed-artifact pipeline; publication and release remain separate."""

import subprocess
import sys
from pathlib import Path

if not Path("results/coding/completion_trace_audit_v1/complete.json").exists():
    raise RuntimeError("Independent final replay incomplete")
for script in (
    "fairness_audit",
    "check_adapter_artifacts",
    "completion_analysis",
    "plot_completion",
    "build_demo",
    "write_completion_report",
):
    subprocess.run([sys.executable, f"scripts/coding/{script}.py"], check=True)
Path("results/coding/completion_analysis_v1/report_ready.json").write_text(
    '{"report_ready":true,"release_frozen":false}'
)
