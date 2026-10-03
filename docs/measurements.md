# Measurements: keep timing scopes separate

All tables are retrospective local regression measurements. The corpus, approved
records, labels and raw traces are private assets, so the public repo provides
scalar aggregates and synthetic contracts rather than a reproducible blind
accuracy dataset. See [original measurements](../verification/local-results.json)
and [Decision 2.0 aggregates](../verification/decision2-results.json).

## Model choice on the same server inputs

RTX 5090 / 32 GiB, sequential resident workers, 2026-10-04:

| Model | Source alignment /150 | General choices /128 | Two-hop goals /6 | Source inference median | GPU peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Laya | 50 | 46 | 1 | 7.9 ms | 2.31 GiB |
| Task-trained Laya | 124 | 57 | 3 | 7.9 ms | 2.31 GiB |
| Decision 2.0 Kai 0.6B | 121 | 98 | 5 | 9.7 ms | 1.50 GiB |
| Decision 2.0 Eos 0.8B | 115 | 94 | 5 | 23.2 ms | 2.07 GiB |
| Decision 2.0 Sol 2B | 129 | 105 | 6 | 24.3 ms | 4.65 GiB |
| Decision 2.0 Nox 4B | 123 | 114 | 6 | 44.8 ms | 9.45 GiB |
| Decision 2.0 Lux 9B | 124 | 116 | 6 | 58.3 ms | 17.04 GiB |

The normal source ranker included a labeled relation in 138/150 menus. Missing
labels were not injected into candidates. General choices are a seeded subset
of 120 graph-evidence rows and 8 tool/routing rows; selecting an accepted first
supporting document is not complete question-answering success. The walk goals
name an intermediate method and use the same two-hop cap and visited exclusion.

No additional training or calibration was performed. Task-trained Laya versus
general Decision checkpoints compares current utility, not equal training budgets.
Native Laya and Decision precision configurations differ. The CUDA run used
Decision BF16 linear residency and FP32 heads; no quantization, ROCm-only kernels
or graph capture. CUDA numbers are not Apple performance estimates.

## Sol on Apple M5 Pro / 48 GiB

Native MPS FP32, PyTorch 2.13.0, Transformers 5.17.0, CPU fallback disabled:

| Task | Result | Median | p95 |
| --- | ---: | ---: | ---: |
| Source alignment | 129/150 | 0.489 s | 0.593 s |
| General choices | 105/128 | 1.021 s | 1.200 s |
| Frozen two-hop state transitions | 6/6 | 1.083 s | 1.102 s |
| Live Neo4j candidate/evidence walk | 6/6 | 1.133 s | 1.284 s |

All 278 single choices matched the server, while maximum probability drift was
about 0.00852. Each walk goal ran three times; live and frozen paths matched.
Cold process/model loading was 12.98 s, first inference 1.88 s, and five total
preparation calls were excluded from warm measurements. Filesystem caches were
not cleared. MPS driver memory reached a sampled 7.93 GiB; tensor allocation and
host RSS overlap in unified memory and must not be added. Sampling does not
guarantee the absolute instantaneous peak.

Single-choice times include tokenization, inference and typed output, excluding
Neo4j reads and a remote agent's final response. Live walk timing includes actual
candidate/proof access and worker round-trips. A unique candidate is followed in
code, so two logical hops can use one or two model calls.

## Whole-agent response time

Matched direct/delegated runs used `gpt-6-luna/high`, one fresh CLI session per
case/arm and alternating order. The direct agent kept one session across hops.
Both arms include CLI initialization, remote turns/tools and the same identifier-
only final response. They do not measure full natural-language synthesis or the
desktop conversation model. Local model and graph startup were excluded.

| Local backend trial | Direct agent median | Agent delegates median | Goal arrival | Reduction |
| --- | ---: | ---: | --- | ---: |
| Laya, six two-hop goals | 15.127 s | 9.897 s | 6/6 direct; 3/6 delegated | 34.6% |
| Sol, same six goals in a separate run | 16.799 s | 13.257 s | 6/6 both | 21.1% |

Laya's trial was 1.53x faster by the ratio of arm medians but lost goal-arrival
quality. Sol's trial was 1.27x faster with 6/6 arrival in both arms. Four Sol
cases were faster and two slower. Remote cache/load/time noise remains; a one-run
small pilot is not a statistical guarantee that every request is faster.

The Sol local tool itself was 1.108 s within the 13.257 s whole-agent duration.
Dividing a 15–17 s whole-agent measurement by a roughly 1.1 s local tool does
not establish a 13–15x whole-response speedup. The earlier ten-case single-source
trial was 12.334 → 9.074 s (1.36x); it is a different task from these two-hop runs.

These data support keeping a replaceable local choice worker and measuring its
latency/goal relevance together. They do not certify unseen navigation, optimal
paths, semantic STOP/BACK, Korean tasks or the scientific truth of graph evidence.
