# Historical HumanEval references

These are historical report values, not newly rerun API models. Our scores use the same benchmark task family but a different instruction prompt and CPython3.14.7 WASI runtime. Neither sampling nor contamination control is matched sufficiently to establish general superiority.

| Historical model | Reported HumanEval score | Reported prompting | Primary source |
|---|---:|---|---|
| GPT-3.5 |48.1%|0-shot|[GPT-4 technical report, Table2](https://cdn.openai.com/papers/gpt-4.pdf)|
| GPT-4 |67.0%|0-shot|[GPT-4 technical report, Table2](https://cdn.openai.com/papers/gpt-4.pdf)|
| Gemini1.0 Pro |67.7%|0-shot, instruction-tuned|[Gemini report v1, Table2 p7](https://arxiv.org/pdf/2312.11805v1)|
| Gemini1.0 Ultra |74.4%|0-shot, instruction-tuned|[Gemini report v1, Table2 p7](https://arxiv.org/pdf/2312.11805v1)|

The GPT report was initially published March2023; exact API checkpoint identifiers and complete HumanEval decoding/sample settings are not specified in that table. The Gemini report v1 is December2023, with table API comparisons collected November2023. Its instruction-tuned flag does not supply a complete replicable prompt/decoding protocol. No temperature, number of samples, parameter count or hidden model revision is inferred from omitted information. Values are treated as published functional-correctness percentages; our measured statistic is one-completion pass@1, and raw numerical differences are descriptive only.

The pinned Gemini v1 PDF SHA-256 is `f3c5b175d26159b92891b32380515ec1036f4ffe2862e19d7277fd165e0d9663`. A current hosted report has later revisions, so v1 is used for historical attribution. The official [HumanEval repository](https://github.com/openai/human-eval) supplies the164 tasks. Pretraining contamination of our open models is unknown. Local split disjointness cannot resolve this limitation.

The earlier0.5B open-model run remains94/164,57.317073% under its recorded local greedy512-token protocol. The final measured1.5B run is reported separately in the machine scorecard after completion. Claims such as “beats GPT-4,” “contamination-free” or “general coding perfection” are not supported by this comparison.
