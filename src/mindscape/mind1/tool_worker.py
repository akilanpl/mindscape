"""Trusted tool dispatcher. Never evaluates submitted Python source in this interpreter."""
import json
import sys


def main():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (12, 13))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8_000_000, 8_000_000))
    except (ImportError, ValueError, OSError):
        pass
    from .engines import run_tool
    data = json.loads(sys.stdin.buffer.read(64001))
    if data["name"] == "research_retrieve":
        from .retrieval import retrieve_arxiv
        retrieved = retrieve_arxiv(**data["args"]).as_dict()
        result = {"ok": retrieved["status"] == "ok", "tool": "research_retrieve", "result": retrieved}
    elif data["name"] == "coding_repository":
        from .repository import analyze_repository
        result = analyze_repository(**data["args"])
    else:
        result = run_tool(data["name"], data["args"])
    output = json.dumps(result, allow_nan=False)
    if len(output) > 128000:
        output = json.dumps({"status": "error", "error": "Tool output limit exceeded"})
    print(output)


if __name__ == "__main__":
    main()
