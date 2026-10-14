# Submitted repository snapshot analysis

`analyze_repository(files: dict[str, str])` analyzes explicitly supplied text.
File paths are labels, never paths to open. No filesystem reads, dependency
installation, execution, subprocesses or network operations occur.

Limits: 64 files, 32,000 UTF-8 bytes per file, 256,000 total bytes, path length
240. Paths must be relative POSIX labels with no traversal, drive markers, empty
components or control characters. Syntax failures are reported per Python file
without aborting the entire snapshot.

Python files receive AST import and named-definition inventories. Cross-file
imports resolve against the submitted root, including bounded relative imports.
Edges remain static candidates: import hooks, package installation, dynamic
imports and alternate source roots are not modeled. Nested function/class names
are inventoried without claiming a complete scope or call graph. Other languages
receive language, byte and line inventory only. Filename extensions can be
ambiguous and do not establish runtime availability.

Snapshots are untrusted data; rendered paths and source text must be escaped.
Public provider integration must not transmit private repository contents without
user authorization. Analysis does not verify architecture or software correctness.

Tests cover cross-file and relative imports, package submodule imports, syntax
errors, nested inventory boundaries, traversal/size rejection and explicit
nonexecution/nonreading of supplied code.
