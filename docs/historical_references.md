# Historical external references: suitability audit

These published general-model references are **not comparable** with our procedural
multiplication task. They are contextual records, not achieved Mindscape targets.
Exact checkpoint hashes are not publicly specified in the cited reports; keep them
unknown rather than guessing an API snapshot.

| Model | Report/version | Dataset/task | Published score | Protocol | Source | Publication date | Suitability |
|---|---|---|---:|---|---|---|---|
| GPT-4 | March 2023 technical report evaluation; exact checkpoint undisclosed | GSM8K, grade-school word problems | 92.0% | 5-shot chain-of-thought; report notes GSM8K training data in pretraining mix | [Table 2 and Appendix E](https://cdn.openai.com/papers/gpt-4.pdf) | Report first submitted 2023-03-15; linked PDF dated 2023-03-27 | Not directly comparable |
| Gemini Ultra | Gemini 1.0, arXiv 2312.11805 v1; exact checkpoint undisclosed | GSM8K, grade-school word problems | 94.4% | Chain-of-thought self-consistency, Table 2 labelled Maj1@32; exact shot/temperature not specified in this table | [Technical report, Table 2](https://arxiv.org/html/2312.11805v1) | 2023-12-19 submission | Not directly comparable |

GPT-4 date metadata: [arXiv record](https://arxiv.org/abs/2303.08774).
Gemini date/version metadata: [v1 record](https://arxiv.org/abs/2312.11805v1).
Checked against primary reports on 2026-10-08. No live proprietary API was used.

## Why a direct claim would be invalid

GSM8K requires understanding word problems and multi-step mathematics. Mindscape's
current inputs are two explicit integers with a hand-specified decimal procedure.
The tasks, inputs, training distributions, decoding and compute differ. Published
pretraining sizes are not a controlled domain-example budget; a ratio against our
1000 multiplication examples would not be a valid DER. Do not place these scores
on our accuracy/data-efficiency plots or claim that multiplication accuracy matches
or beats either system. The two historical protocols also differ from each other.

## Methodology for a future genuine comparison

Choose a shared task/dataset/split first. Freeze interpretation, scoring, prompt
examples, decoding/sample count, tool access and evaluation budget. Verify that
Mindscape actually supports the task; do not carve an easier subset after seeing
results. Cite exact published revision/protocol, record any unknown checkpoint or
settings, audit contamination and separate reference-only from reproduced results.
If exact protocol compatibility cannot be established, report both contextual values
with explicit differences and make no superiority or training-data-efficiency claim.
Use the internal four-condition controlled comparison as the primary evidence.
