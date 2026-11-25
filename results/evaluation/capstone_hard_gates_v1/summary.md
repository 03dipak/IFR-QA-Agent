# Capstone hard-gates evaluation summary

- Run ID: `capstone_hard_gates_v1`
- Mode: `offline_oracle_standin; strategy prompts and gold evidence reused; no live model calls`
- Gold: `data/eval/datasets/gold_v2_eval_v8_public.jsonl`; sha256 `828b6704037297cd2e86da4c514edd9665b1945ee9c062ae7e5a15e417f0b09c`
- Split: `dev`; rows `73`; holdout rows excluded `24`
- Corpus freeze: `410d5781cbbd5aeade55c7d77c0e6b1d517738b0214c56f5bb9879b2b533044b`
- Cost proxy: `college endpoint: no published price; cost proxy is tokens only`
- Caveat: offline oracle stand-in validates the harness and report contracts; it is not live model accuracy.

## Summary metrics

| Metric | Value | Denominator |
| --- | ---: | ---: |
| End-to-end accuracy | 1.0 | 73 |
| Answer accuracy | 1.0 | 45 |
| Abstain accuracy | 1.0 | 28 |
| Task completion rate | 1.0 | 73 |
| Avg tool calls/query | 0.6164 | 73 |
| Cost proxy/query USD | 0.0 | 73 |
| Latency ms/query | 8.0822 | 73 |
| Mean tokens/query | 29.2603 | 73 |

## Retrieval recall@k (reported separately)

| Metric | Hits | Denominator | Value |
| --- | ---: | ---: | ---: |
| recall@1 | 45 | 46 | 0.9783 |
| recall@3 | 45 | 46 | 0.9783 |
| recall@5 | 45 | 46 | 0.9783 |
| recall@10 | 45 | 46 | 0.9783 |

## Failure taxonomy

| Label | Count | Denominator | Rate |
| --- | ---: | ---: | ---: |
| retrieval_miss | 0 | 73 | 0.0 |
| reasoning_error | 0 | 73 | 0.0 |
| tool_call_error | 0 | 73 | 0.0 |
| hallucination | 0 | 73 | 0.0 |

## Baseline vs agent (same run)

| Arm | Source | n | Accuracy | Completion | Tool calls | Cost/query | Latency ms/query | Recall@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | `offline_oracle_standin; strategy prompts and gold evidence reused; no live model calls` | 73 | 0.8493 | n/a | 0.0 | 0.0 | 0.0 | 0.9783 |
| S2 | `offline_oracle_standin; strategy prompts and gold evidence reused; no live model calls` | 73 | 1.0 | n/a | 0.0 | 0.0 | 0.0 | 0.9783 |
| S3 | `offline_oracle_standin; strategy prompts and gold evidence reused; no live model calls` | 73 | 1.0 | n/a | 0.0 | 0.0 | 0.0 | 0.9783 |
| AGENT | `offline_oracle_standin; strategy prompts and gold evidence reused; no live model calls` | 73 | 1.0 | 1.0 | 0.6164 | 0.0 | 8.0822 | 0.9783 |

## Paired comparison counts

| Arm | Both passed | Both failed | Agent only | Baseline only |
| --- | ---: | ---: | ---: | ---: |
| S1 vs AGENT | 62 | 0 | 11 | 0 |
| S2 vs AGENT | 73 | 0 | 0 | 0 |
| S3 vs AGENT | 73 | 0 | 0 | 0 |

Report lint: `True`; errors `[]`
Document-to-questions index documents: `38`
