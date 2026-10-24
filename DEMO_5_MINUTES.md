# Five-minute evaluation demonstration

Before entering the venue: charge the Mac, close heavy applications, retain the downloaded model weights, launch START_MINDSCAPE.command, open the loopback URL, confirm neural.ready=true, and run REHEARSE.command. This uses local CPU and cached weights, not Kaggle. Disable Wi-Fi for the offline rehearsal after dependencies are installed. If model initialization fails, use START_TOOLS_BACKUP.command and explicitly disclose solver-only operation.

0:00–0:30 — Explain the architecture. Mindscape combines a single pretrained Qwen model with verified computation. The custom native R6 network is separate research and did not learn these tool capabilities.

0:30–1:20 — Ask: Differentiate x^3 + sin(x) with respect to x. Show the interpreted problem, symbolic answer and worked steps. Open View activity and show pretrained interpretation, supplied grammar/grounding and exact tool provenance. Avoid describing an advisory neural plan as a mathematical proof.

1:20–2:10 — Ask: Find current through a 4 ohm resistor with 12 V across it. Show I=V/R,3 A and explicit dimensional checking. If optics is in the accepted scope, optionally show Lens equation, f=20 cm and u=-30 cm, emphasizing Cartesian signs.

2:10–3:00 — Ask Predict the output with this exact code:

```python
def total(n):
    s=0
    for i in range(n):
        s+=i
    return s
print(total(5))
```

Show10 and the bounded AST trace. It is an interpreter-backed result, not evidence that R6 learned Python execution.

3:00–3:35 — Ask a request outside the accepted scope (for example a missing-data circuit). Show explicit unsupported handling instead of fabricated certainty. If time permits demonstrate SELECT with an explicitly supplied table.

3:35–4:25 — Open Model & research, then Run frozen native R6 demonstration. Show the54,014-parameter custom neural graph architecture and actual preserved checkpoint predictions alongside external reference answers. Acknowledge arithmetic/coding collapse and weak physics OOD. Native CBSE natural-language scores are unsupported, not zero or hybrid scores.

4:25–5:00 — Present the held-out acceptance summary and limitations. Scores apply to this small frozen question set; topics missing the predeclared threshold are excluded from guaranteed demo scope. Pretrained-only accuracy, hybrid accuracy, correctness of explanations and latency are reported separately. No broad Class12 mastery claim. State that no new training or paid resources were used.
