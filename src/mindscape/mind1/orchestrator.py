"""Bounded tool/model orchestration with recorded evidence, not hidden reasoning."""
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

MODES = {"quick": (1, 20), "standard": (3, 60), "deep": (6, 100), "research": (8, 120)}


def classify(message):
    if re.search(r'```|\b(code|debug|python|rust|javascript|compiler|algorithm|repository|sql|cuda)\b', message, re.I):
        return "coding"
    if re.search(r'\b(physics|force|mass|gravity|quantum|relativity|projectile|oscillator|circuit|lagrangian|energy)\b', message, re.I):
        return "physics"
    return "math"


def validate(data):
    if not isinstance(data, dict):
        raise ValueError("JSON object required")
    message = data.get("message")
    if not isinstance(message, str) or not message.strip() or len(message) > 16000:
        raise ValueError("Message must contain 1–16000 characters")
    mode, domain = data.get("mode", "standard"), data.get("domain", "auto")
    if not isinstance(mode, str) or not isinstance(domain, str) or mode not in MODES or domain not in {"auto", "math", "coding", "physics"}:
        raise ValueError("Unknown reasoning mode or domain")
    history = data.get("history", [])
    if not isinstance(history, list) or len(history) > 40:
        raise ValueError("History must be a list of at most 40 messages")
    clean = []
    for item in history:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str) or item.get("role") not in {"user", "assistant"} or not isinstance(item.get("content"), str) or len(item["content"]) > 16000:
            raise ValueError("Only bounded user/assistant text history is accepted")
        clean.append({"role": item["role"], "content": item["content"]})
    if sum(len(x["content"]) for x in clean) > 64000:
        raise ValueError("Conversation too large")
    return message.strip(), mode, classify(message) if domain == "auto" else domain, clean


def execute_tool(name, args, cancel=None, timeout=15):
    if not isinstance(name, str) or not isinstance(args, dict):
        raise ValueError("Invalid tool call")
    payload = json.dumps({"name": name, "args": args})
    if len(payload.encode()) > 64000:
        raise ValueError("Tool arguments exceed limit")
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "SYSTEMROOT", "MIND1_WASI_RUNTIME", "MINDSCAPE_WASI_RUNTIME", "MIND1_RETRIEVAL", "MIND1_NATIVE_CHECKPOINT", "MIND1_NATIVE_CHECKPOINT_SHA256"}}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    with subprocess.Popen([sys.executable, "-m", "mindscape.mind1.tool_worker"], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env) as process:
        start = time.monotonic()
        first = True
        while True:
            if cancel and cancel.is_set():
                process.kill(); process.communicate()
                raise InterruptedError("Cancelled")
            remaining = timeout - (time.monotonic()-start)
            if remaining <= 0:
                process.kill(); process.communicate()
                return {"ok": False, "error": {"code": "timeout", "message": "Tool wall time exceeded"}}
            try:
                out, err = process.communicate(payload.encode() if first else None, timeout=min(.1, remaining))
                break
            except subprocess.TimeoutExpired:
                first = False
        if process.returncode or len(out) > 128000:
            return {"ok": False, "error": {"code": "worker_failure", "message": "Tool failed or exceeded output limit"}}
        try:
            result = json.loads(out)
            if not isinstance(result, dict):
                raise ValueError()
            return result
        except (ValueError, TypeError):
            return {"ok": False, "error": {"code": "invalid_output", "message": "Invalid tool response"}}


def local_plan(message):
    """Conservative offline parsing. This parser is supplied grammar, not learned language understanding."""
    text = message.strip().rstrip("?.")
    block = re.search(r'```([a-zA-Z+#]*)\n(.*?)```', message, re.S)
    if block and re.match(r"^\s*(?:inspect|analyze|analyse|check)\b", message, re.I):
        lang = block[1].lower() or "python"
        lang = {"py": "python", "js": "javascript", "ts": "typescript", "c++": "cpp"}.get(lang, lang)
        return "coding_analyze", {"source": block[2], "language": lang}
    match = re.fullmatch(r'(?:what is |calculate |evaluate |compute )?([\d\s.+*/()^%-]+)', text, re.I)
    if match:
        return "math_calculate", {"expression": match[1], "operation": "evaluate"}
    match = re.fullmatch(r'(differentiate|integrate|simplify|solve)\s+(.+?)(?:\s+(?:with respect to|for)\s+([a-z]))?', text, re.I)
    if match:
        expr = match[2]
        if match[1].lower() == "solve" and "=" in expr:
            left, right = expr.split("=", 1)
            expr = f"({left})-({right})"
        return "math_calculate", {"expression": expr, "operation": match[1].lower(), "variable": match[3] or "x"}
    return None


class Mind1Orchestrator:
    def __init__(self, runner=execute_tool, runtime=None):
        from .native_runtime import NativeMindscapeRuntime
        self.runner = runner
        self.runtime = runtime or NativeMindscapeRuntime(runner)

    def capabilities(self):
        from .engines import capabilities
        return {"product": "Mind 1", "native": self.runtime.capability(), "tools": capabilities(),
                "modes": list(MODES), "r56": "R5.6 EXPERIMENTAL — SCIENTIFIC GENERALIZATION UNVERIFIED",
                "external_inference": False,
                "retention": "Browser-local history; backend does not persist conversations",
                "public_accounts": "Unavailable: loopback application only"}

    def chat(self, data, cancel=None):
        from .language import interpret
        message, mode, domain, history = validate(data)
        if type(data.get('native_demo', False)) is not bool:
            raise ValueError('native_demo must be a boolean')
        cancel = cancel or threading.Event()
        if cancel.is_set():
            raise InterruptedError('Cancelled')
        began = time.monotonic()
        parsed = interpret(message, history=history)
        trace = [{"kind": "interpretation", "summary": "Bounded deterministic native grammar", "representation": parsed}]
        tools, limitations = [], list(parsed.get('limitations', []))
        provenance = {"product": "Mind 1", "external_inference": False, "r56_used": False,
                      "scientific_status": "R5.6 EXPERIMENTAL — SCIENTIFIC GENERALIZATION UNVERIFIED"}
        native = None
        problem = dict(parsed.get('problem') or {})
        concept_ids = {'add': [0], 'add_digit': [0], 'borrow_digit': [1], 'rename': [3], 'force': [2]}
        if parsed.get('kind') == 'native':
            problem.setdefault('concept_ids', concept_ids.get(problem.get('operation'), []))
            trace.append({'kind': 'concepts', 'summary': 'Researcher-supplied concept IDs; embedding parameters are untrained', 'concept_ids': problem['concept_ids']})
        if parsed.get('kind') == 'native':
            domain = problem.get('domain', domain)
            native = self.runtime.run(problem, engineering_demo=data.get('native_demo', False), cancel=cancel, timeout=15)
            trace.append({"kind": "native_prediction", "summary": "R5.6 original prediction and post-hoc verification", "record": native})
            if native.get('available'):
                provenance.update(r56_used=True, checkpoint=native.get('checkpoint'), attribution=native.get('attribution'))
                label = "trained but unqualified" if native.get("checkpoint", {}).get("trained") else "untrained"
                answer = f"ENGINEERING DEMONSTRATION — {label} native R5.6 weights. This is not qualified intelligence.\n\nOriginal neural prediction:\n```json\n" + json.dumps(native.get('prediction'), indent=2) + "\n```\n\nIndependent verification (prediction unchanged):\n```json\n" + json.dumps(native.get('verification'), indent=2) + "\n```"
                limitations.append('No trained or scientifically qualified native model; a passing individual check is not generalization evidence')
                return self._result(answer, domain, trace, provenance, limitations, tools, began, 'engineering_demo', native)
            limitations.append(native.get('reason', 'Native checkpoint unavailable'))
        planned = parsed.get('tool')
        if not planned:
            fallback = local_plan(message)
            if fallback:
                planned = {'name': fallback[0], 'args': fallback[1]}
        # Supported arithmetic still has an independently attributed exact tool path.
        if not planned and parsed.get('kind') == 'native' and domain == 'math' and problem.get('operation') == 'add':
            planned = {'name': 'math_calculate', 'args': {'expression': str(problem['left']) + '+' + str(problem['right'])}}
        if planned:
            output = self.runner(planned['name'], planned['args'], cancel=cancel, timeout=15)
            tools.append({'name': planned['name'], 'arguments': planned['args'], 'output': output})
            trace.append({'kind': 'tool', 'summary': planned['name'] + (': completed' if output.get('ok') else ': failed'), 'result': output})
            provenance['attribution'] = 'Deterministic scientific tool; not learned reasoning'
            if output.get('ok'):
                answer = 'Verified tool pathway (not neural intelligence).\n\n```json\n' + json.dumps(output.get('result'), indent=2) + '\n```'
                if native is not None:
                    answer += '\n\nNative R5.6 status: no qualified checkpoint. An explicitly labeled engineering demo is available.'
                return self._result(answer, domain, trace, provenance, limitations, tools, began, 'completed', native)
            limitations.append('Tool operation failed; no result is certified')
        answer = 'This question is outside the currently supported native interpretation or qualified model capability. No external model is used. R5.6 has no qualified learned checkpoint. Supported tools can still perform explicit calculations, symbolic calculus and safe code inspection. No solution is claimed.'
        return self._result(answer, domain, trace, provenance, limitations, tools, began, 'unavailable', native)

    @staticmethod
    def _result(answer, domain, trace, provenance, limitations, tools, began, status, native=None):
        provenance['runtime_seconds'] = round(time.monotonic()-began, 3)
        return dict(answer=answer, domain=domain, trace=trace, provenance=provenance,
                    limitations=limitations, tools=tools, status=status, native=native)


Orchestrator = Mind1Orchestrator
