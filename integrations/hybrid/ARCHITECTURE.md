# Mindscape architecture and limits

The functional assistant uses Qwen/Qwen3-0.6B, a separately pretrained open-weight general model, for an advisory structured interpretation. A conservative supplied grammar or literal-grounding check determines which model fields may reach a solver. The UI displays the exact interpreted problem. Parser rules are engineered software, not learned neural competence.

The deterministic tools handle restricted-AST SymPy calculus/matrices/polynomials, explicitly dimensional physics formulas, a bounded Python AST interpreter and read-only SQLite over user-provided rows. A separate subprocess limits computation. Worked steps are generated from those actual calculations. Tool checks sometimes reuse the same symbolic engine; the independent held-out evaluator supplies a separate answer/reference check. An explanation is not deemed correct merely because an answer matches.

The original Mind1 browser interface now calls the local /api/chat endpoint. Modes are hybrid, pretrained-only and tools-only. Every answer identifies its actual method. A native-research route declines general CBSE input, while a separate button actually loads the preserved R6 checkpoint on CPU and demonstrates its supported schema and failures.

Native R6 is a54,014-parameter custom network with character GRU encoding/decoding and typed property-node message passing for physics. Its arithmetic/coding collapsed, and average physics improvements did not establish physical-law generalization. R6.1's masked pooling improves gradient access without proven learned recovery; its proposed training is unapproved and has not run. Pretrained/tool capabilities are not attributed to either native model.

Limitations include single-model small-capacity parsing, restricted natural-language grammar, limited solver domains, shared-account/trusted-controller evaluation ownership, small topic samples, no public hostile-code sandbox claim and no broad CBSE mastery. The venue mode is local CPU with cached weights. A solver-only local backup is available and must be labeled as such. No paid service or public deployment is used.
