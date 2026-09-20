# EvalForge — Experimentation Program & Findings

## 1. Purpose

This document consolidates the experimentation performed during the EvalForge development sprint.

The objective was not simply to maximize an isolated retrieval metric. The experimentation program was designed to determine:

- how different retrieval strategies affect evidence retrieval,
- whether reranking improvements propagate into final LLM answers,
- whether context ordering itself affects generation,
- where retrieval failures occur,
- which cases are useful for deeper diagnosis,
- and whether retrieval improvements translate into end-to-end RAG quality.

The experiments were deliberately separated from the production EvalForge pipeline and are preserved under `experiments/`.

---

# 2. Experimental Baseline

## Corpus

- 16 source documents
- 17 chunks in the baseline corpus
- 75 evaluation cases for the primary benchmark

## Baseline retrieval

The baseline dense retriever used:

- SentenceTransformers
- `BAAI/bge-small-en-v1.5`
- 384-dimensional embeddings
- top-k retrieval

The benchmark also evaluated BM25, hybrid retrieval, and CrossEncoder reranking.

## Evaluation dimensions

Retrieval experiments measured:

- Hit@1
- Hit@3
- Hit@5
- Recall@1
- Recall@3
- Recall@5
- MRR
- retrieval latency

End-to-end experiments additionally evaluated:

- answer correctness
- groundedness
- topic coverage
- missing topics
- unsupported claims
- generation latency
- token usage
- estimated cost
- structured-output validity

---

# 3. Retrieval Strategy Experiment — B1

## Question

How do dense retrieval, BM25, hybrid retrieval, and CrossEncoder reranking compare on the 75-case benchmark?

## Results

| Strategy | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | MRR | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.8667 | 0.9733 | 0.9867 | 0.7889 | 0.9489 | 0.9800 | 0.9189 | ~21 ms |
| BM25 | 0.8400 | 0.9600 | 0.9867 | 0.7778 | 0.9222 | 0.9600 | 0.9022 | ~1.8 ms |
| Hybrid | 0.8800 | 0.9867 | 1.0000 | 0.8089 | 0.9600 | 0.9889 | 0.9322 | ~23 ms |
| Dense + CrossEncoder | 0.9333 | 1.0000 | 1.0000 | 0.8556 | 0.9733 | 0.9956 | 0.9644 | ~351 ms |
| Hybrid + CrossEncoder | 0.9333 | 1.0000 | 1.0000 | 0.8556 | 0.9667 | 0.9956 | 0.9644 | ~478 ms |

## Finding

CrossEncoder reranking produced the strongest retrieval ranking on this benchmark.

Dense + CrossEncoder was selected as the experimental reranking configuration because it achieved the same primary quality figures as Hybrid + CrossEncoder while having lower measured latency.

This is a **benchmark-specific engineering decision**, not a claim that dense + CrossEncoder is universally superior.

---

# 4. CrossEncoder Forensic Analysis

A dedicated forensic comparison examined how CrossEncoder reranking changed the dense top-5 results.

## Classification

Across 75 cases:

- NEUTRAL_REORDER: 52
- RANKING_IMPROVEMENT: 9
- UNCHANGED: 13
- RANKING_REGRESSION: 1

The CrossEncoder therefore changed ordering frequently, but usually did not change which documents were present.

## Retrieval impact

Dense:

- Hit@1: 0.8667
- Hit@3: 0.9733
- Hit@5: 0.9867
- Recall@1: 0.7889
- Recall@3: 0.9489
- Recall@5: 0.9800
- MRR: 0.9189

CrossEncoder:

- Hit@1: 0.9467
- Hit@3: 0.9867
- Hit@5: 0.9867
- Recall@1: 0.8689
- Recall@3: 0.9644
- Recall@5: 0.9800
- MRR: 0.9644

Delta:

- Hit@1: +0.08
- Hit@3: +0.0133
- Hit@5: 0
- Recall@1: +0.08
- Recall@3: +0.0156
- Recall@5: 0
- MRR: +0.0456

## Case transitions

- 64 PASS → PASS
- 6 RANKING_FAILURE → PASS
- 1 MULTI_EVIDENCE_RANKING_FAILURE → PASS
- 1 PASS → MULTI_EVIDENCE_RANKING_FAILURE
- 1 PARTIAL_RECALL → PARTIAL_RECALL
- 1 RETRIEVAL_FAILURE → RETRIEVAL_FAILURE
- 1 RANKING_FAILURE → RANKING_FAILURE

Seven cases were fully fixed at the retrieval-ranking level, two improved partially while remaining failures, and one regressed.

## Important regression

`routing_006` demonstrated a multi-evidence ranking conflict.

Dense placed `model-routing.md` first and achieved MRR 1.0.

CrossEncoder placed `deployment-policy.md` first, moving `model-routing.md` to rank 2. MRR dropped to 0.5.

The expected evidence remained available. The failure was therefore about ranking and multi-evidence coverage rather than evidence availability.

---

# 5. Candidate-K Experiment

## Question

How much retrieval quality is gained by increasing the candidate pool before reranking?

Candidate pools of 5, 10, 15, and 17 were compared.

## Results

| Candidate K | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | MRR | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K=5 | 0.9467 | 0.9867 | 0.9867 | 0.8689 | 0.9644 | 0.9800 | 0.9644 | ~159 ms |
| K=10 | 0.9333 | 1.0000 | 1.0000 | 0.8556 | 0.9733 | 0.9956 | 0.9644 | ~386 ms |
| K=15 | same as K=10 | same | same | same | same | same | same | ~569 ms |
| K=17 | same as K=10 | same | same | same | same | same | same | ~650 ms |

67/75 final ranked lists differed between K=5 and K=10, but only two cases gained expected evidence from the larger candidate pool.

Performance plateaued beyond approximately K=10.

## Finding

For the current benchmark, K=5 was selected as the operating point because it retained the same MRR as larger pools while substantially reducing latency.

A retrieval implementation bug was also discovered and fixed: an explicit `candidate_k` had previously been silently expanded to the reranker candidate pool. The corrected behavior makes explicit candidate K authoritative, expanding only when necessary for the requested top-k output.

---

# 6. Chunking Ablation

## Question

How sensitive is retrieval to chunk size and overlap?

## Chunk-size results

| Configuration | MRR | Hit@1 | Recall@5 |
|---|---:|---:|---:|
| 120 / 30 baseline | 0.9189 | 0.8667 | 0.9800 |
| 60 / 15 | 0.9071 | 0.8533 | 0.9689 |
| 240 / 60 | 0.9256 | 0.8800 | 0.9800 |

The 240/60 configuration showed a small MRR/Hit@1 improvement, while Recall@5 remained unchanged.

## Overlap sweep

The overlap sweep tested:

- 120/0
- 120/15
- 120/30
- 120/60

Aggregate quality remained very similar.

The experiment changed rankings in some cases but did not demonstrate a sufficiently strong benchmark-wide improvement to justify changing the production configuration.

## Finding

The evidence did not justify changing the production chunking configuration during this sprint.

The experiment also demonstrated that a ranking improvement does not necessarily represent increased evidence recall.

---

# 7. G1 — Retrieval → Generation Propagation

## Question

Does improved retrieval ranking actually improve the final LLM answer?

Two conditions were compared over the full 75-case benchmark:

1. Dense → top-5 → LLM
2. Dense → CrossEncoder → top-5 → LLM

The same model, generation settings, prompts, dataset, and judge were used.

Generation was performed in batches of five cases per request to avoid the Gemini free-tier request-per-minute limitation encountered during individual generation.

## Results

### Dense

- Answer correctness: 75/75
- Groundedness: 75/75
- Unsupported claims: 0/75
- Mean topic coverage: ~97.87%

### Dense + CrossEncoder

- Answer correctness: 73/75
- Groundedness: 75/75
- Unsupported claims: 0/75
- Mean topic coverage: ~97.87%

## Propagation

- Retrieval ranking improvement → answer improvement: 0
- Retrieval ranking improvement → answer unchanged: 9
- Retrieval ranking regression → answer unchanged: 1
- Neutral reorder → answer unchanged: 50
- Neutral reorder → answer regression: 2
- Unchanged retrieval → answer unchanged: 13

Overall:

- 0 answer improvements
- 73 unchanged
- 2 regressions

## Important finding

Improved retrieval metrics did **not** automatically produce better final answers.

All nine genuine retrieval ranking improvements were downstream-neutral.

The single retrieval ranking regression was also downstream-neutral.

The two answer regressions occurred during neutral reordering where the evidence set itself remained available.

This established a key EvalForge finding:

> Retrieval quality improvements are not sufficient evidence of end-to-end RAG improvement.

Context ordering can influence generation even when the evidence set is unchanged.

---

# 8. G2 — Context Ordering

## Question

Can context ordering itself explain changes in generation?

Four conditions were compared:

1. Dense
2. Dense + CrossEncoder
3. Dense evidence with the first expected document moved to rank 1
4. Oracle ordering, where expected documents were moved to the front in expected-document order

The expected-first and oracle conditions did not add evidence. They only reordered evidence already present in the Dense top-5.

## Findings

The experiment separated:

- evidence availability,
- evidence ordering,
- and downstream synthesis.

The benchmark was highly answerable: ordering interventions generally did not change answer correctness.

Two diagnostic cases were particularly important:

### `cost_005`

The evidence set supported the answer under Dense and CrossEncoder ordering.

An oracle ordering caused the generated answer to omit the required synthesis relationship between model quality, latency, and cost.

The answer remained grounded, but topic coverage became incomplete.

### `golden_004`

The expected `golden-datasets.md` evidence was absent from both retrieval conditions.

Dense nevertheless produced a correct answer from secondary evidence.

A different ordering produced an incorrect answer while remaining grounded.

## Finding

A correct RAG answer does not necessarily require retrieval of the exact expected document, and a better evidence ordering does not guarantee a better answer.

This reinforced the distinction between:

- retrieval recall,
- evidence ranking,
- and semantic synthesis.

---

# 9. G3A — Failure Mining

## Purpose

G3A mined the 75-case benchmark to identify cases where retrieval behavior could plausibly affect final answer quality.

The analysis classified cases using retrieval changes and end-to-end answer outcomes.

## Results

Across the full benchmark:

- retrieval-sensitive cases: none
- retrieval-insensitive cases: 73
- oracle-paradox cases: 2

The two oracle-paradox cases were:

- `cost_005`
- `golden_004`

## Finding

The original 75-case benchmark was highly answerable at the generation layer.

This meant that many retrieval improvements were masked by the LLM's ability to synthesize a correct answer from secondary or already-sufficient evidence.

G3 therefore shifted from broad benchmark evaluation toward targeted diagnostic cases.

---

# 10. G3B — Candidate Screening and End-to-End Diagnostic Set

## Retrieval-only screening

A 16-case screening set was used to identify cases with meaningful ranking changes.

Results:

- Dense Hit@1: 0.4375
- Dense Hit@3: 0.875
- Dense Hit@5: 0.9375
- Dense Recall@5: 0.90625
- Dense MRR: 0.6510

CrossEncoder:

- Hit@1: 0.75
- Hit@3: 0.9375
- Hit@5: 0.9375
- Recall@5: 0.90625
- MRR: 0.8333

Across all 16 cases:

- document-set changes: 0
- order changes: 16
- expected-rank changes: 13
- expected-document presence changes: 0

## Candidate selection

The strongest diagnostic candidates included:

- `cross_003`
- `routing_006`
- `evaluation_strategy_003`
- `golden_004`
- `context_004`
- `deployment_004`
- `embedding_004`
- `retrieval_004`
- `validation_005`
- `reliability_002`

A 10-case end-to-end diagnostic experiment was run.

## G3B end-to-end result

Both conditions achieved:

- Dense: 10/10 correct, 10/10 grounded
- CrossEncoder: 10/10 correct, 10/10 grounded

Propagation:

- answer improvements: 0
- unchanged: 10
- regressions: 0

## Finding

Even deliberately selected retrieval-sensitive cases did not guarantee downstream answer sensitivity.

This showed that candidate mining needed to become more targeted rather than simply selecting cases with large retrieval metric changes.

---

# 11. G3C — Final Retrieval-Dependent Checkpoint

G3C was the final diagnostic experiment in the experimentation program.

No further G4 experiment was pursued.

## Retrieval checkpoint

The final 10-case candidate pool was reduced to an 8-case diagnostic set:

- `cross_003`
- `routing_006`
- `cross_001`
- `evaluation_strategy_003`
- `deployment_004`
- `embedding_004`
- `context_004`
- `cost_005`

Roles included:

- strong ranking improvement
- ranking regression
- three-document synthesis
- missing-evidence control
- multi-document ranking improvement
- smaller ranking improvement
- pure expected-evidence reorder
- retrieval-invariant control

The retrieval checkpoint showed:

- document-set changed cases: 0/10
- order changed cases: 9/10
- expected rank changed cases: 7/10
- missing expected evidence changed cases: 0/10

This confirmed that the selected G3C experiment primarily studied **ranking and ordering**, rather than evidence-set changes.

## Final valid G3C end-to-end run

The corrected G3C v2 experiment supplied the **actual retrieved document text** to the LLM and converted retrieved evidence into the proper `RetrievedEvidence` objects for judging.

This correction is important because the earlier G3C v1 artifact did not pass the actual evidence text into generation and therefore was not a valid RAG propagation experiment.

The valid artifact is:

`g3c_end_to_end_v2.json`

## Results

| Condition | Correct | Grounded |
|---|---:|---:|
| Dense | 7/8 (87.5%) | 7/8 |
| Dense + CrossEncoder | 8/8 (100%) | 8/8 |

Propagation:

- ANSWER_IMPROVEMENT: 1
- ANSWER_UNCHANGED: 7
- ANSWER_REGRESSION: 0

## Key result

G3C finally produced a case where improved retrieval ranking propagated into final answer quality.

The improvement occurred on `cross_001`.

The Dense condition had the required evidence available but generated an insufficient answer. The CrossEncoder condition reordered the same evidence and produced the expected answer.

This provides direct evidence that:

> Retrieval ranking can affect final answer quality even when the retrieved document set does not change.

At the same time, the experiment is an **8-case targeted diagnostic**, not a replacement for the 75-case benchmark.

---

# 12. Gemini Rate-Limit Finding

During the experimentation program, individual generation against the Gemini free tier encountered:

`429 Too Many Requests`

The reported quota was:

`GenerateRequestsPerMinutePerProjectPerModel-FreeTier`

with a quota value of approximately 15 requests per minute.

A 75-case individual generation workload therefore produced rate-limit failures.

## Batching experiment

### 5 cases/request

- 75/75 cases
- 15 requests
- 0 integrity failures
- exact count/order preserved
- ~36.88 s wall time
- 80% request reduction

### 15 cases/request

- 75/75 cases
- 5 requests
- 0 integrity failures
- exact count/order preserved
- ~13.51 s wall time
- 93.3% request reduction

A 5-case batch was ultimately selected for the production-style experimentation harness because it substantially reduced request count while keeping the batch size manageable and compatible with strict integrity validation.

The project deliberately did not switch to the Gemini Batch API.

---

# 13. Engineering Lessons

## 13.1 Retrieval metrics are necessary but insufficient

CrossEncoder reranking materially improved retrieval metrics:

- MRR increased from 0.9189 to 0.9644
- Hit@1 increased from 0.8667 to 0.9467

But G1 showed that these gains did not automatically improve answer correctness.

Therefore EvalForge must distinguish:

`retrieval quality`

from

`end-to-end RAG quality`.

---

## 13.2 Ranking and evidence recall are different problems

Several experiments showed that CrossEncoder reranking mostly changed the order of an already-present evidence set.

In the 16-case screening:

- document-set changed: 0
- ordering changed: 16

Therefore a reranker cannot recover evidence that the first-stage retriever never retrieved.

---

## 13.3 Multi-evidence questions expose ranking weaknesses

`routing_006` demonstrated that placing one highly relevant document above another can hurt a question requiring multiple pieces of evidence.

A reranker optimized for pairwise relevance may therefore conflict with multi-document synthesis requirements.

---

## 13.4 LLMs can compensate for retrieval failures

`golden_004` demonstrated that the expected document can be absent while the LLM still produces a correct answer using secondary evidence.

Therefore:

`retrieval failure != necessarily answer failure`

and:

`retrieval success != necessarily answer success`.

---

## 13.5 Context order matters

G1 and G2 showed that changing ordering while preserving the evidence set can alter the final answer.

This means context construction is itself an experimental variable.

---

## 13.6 Benchmark difficulty matters

The original 75-case benchmark was too easy at the answer layer for many retrieval improvements to propagate.

The G3 diagnostic process was therefore necessary to expose cases where retrieval differences had a realistic chance of affecting generation.

---

## 13.7 Experiment integrity matters

The G3C v1 failure exposed an important experimental-design lesson.

Retrieval metadata such as:

```text
rank
document_id
chunk_id
```

is not equivalent to providing the actual retrieved evidence.

A valid RAG propagation experiment must pass the actual retrieved text into generation and the same evidence representation into judging.

The v2 rerun corrected this.

---

# 14. Final Experimental Conclusions

The experimentation program established the following evidence-backed conclusions for EvalForge:

### Retrieval

1. Dense retrieval provides a strong baseline on the current corpus.
2. BM25 is extremely fast but slightly weaker on the benchmark.
3. Hybrid retrieval improves retrieval metrics over dense retrieval.
4. CrossEncoder reranking produces the strongest measured ranking quality on the benchmark.
5. The CrossEncoder's primary effect is often reordering existing evidence rather than adding missing evidence.
6. Larger reranking candidate pools have diminishing returns beyond approximately K=10.
7. K=5 provides a useful latency/quality operating point for the current benchmark.
8. Chunking changes retrieval ranking, but the current evidence did not justify changing the production chunking configuration.

### Generation

9. Retrieval improvements do not automatically propagate into better answers.
10. LLM generation can remain correct despite retrieval ranking regressions.
11. LLM generation can fail despite apparently adequate retrieval.
12. Context ordering can affect answer quality even when the evidence set is unchanged.
13. Multi-document synthesis is a particularly important failure mode.

### Evaluation

14. A serious RAG evaluation system must evaluate both retrieval and generation.
15. Retrieval metrics alone cannot establish end-to-end RAG quality.
16. Targeted diagnostic cases are valuable when a broad benchmark is too answerable.
17. Structured-output validation and case-level integrity checks are essential for batch experiments.
18. Checkpointing and retry handling are necessary for long-running LLM evaluations under provider rate limits.

### Engineering

19. Batch generation substantially reduces provider request pressure.
20. Experiment artifacts should remain isolated from production code.
21. Experimental conclusions should be tied to measured benchmark conditions rather than generalized beyond the tested corpus and dataset.

---

# 15. Final Experimentation Status

| Experiment | Status | Purpose |
|---|---|---|
| B1 Retrieval Strategy Matrix | Complete | Compare retrieval strategies |
| Candidate-K | Complete | Find reranking candidate operating point |
| Chunking Ablation | Complete | Test chunk-size/overlap sensitivity |
| CrossEncoder Forensics | Complete | Analyze ranking changes |
| G1 Retrieval → Generation | Complete | Measure downstream propagation |
| G2 Context Ordering | Complete | Isolate ordering effects |
| G3A Failure Mining | Complete | Mine retrieval-sensitive failures |
| G3B Candidate Screening | Complete | Target diagnostic cases |
| G3B End-to-End | Complete | Test selected diagnostic cases |
| G3C Candidate Mining | Complete | Final targeted candidate selection |
| G3C Retrieval Checkpoint | Complete | Validate final diagnostic set |
| G3C End-to-End v2 | Complete | Final retrieval-dependent experiment |

**Experimentation track: CLOSED.**

No G4+ experiment is required for the current EvalForge sprint.

---

# 16. Final Artifacts

The experiment artifacts are isolated under:

```text
experiments/
├── retrieval/
├── g1/
├── g2/
├── g3/
└── legacy/
```

The final G3C artifact is:

```text
experiments/g3/results/g3c_end_to_end_v2.json
```

The earlier G3C v1 artifact is retained as experimental provenance but should not be used for final conclusions because it did not provide actual retrieved evidence text to generation.

---

# 17. One-Line Project Finding

> **EvalForge demonstrated that improving retrieval ranking is measurable and valuable, but retrieval metrics alone do not establish better RAG behavior; end-to-end evaluation revealed that evidence ordering, multi-document synthesis, and LLM compensation for retrieval failures can determine whether retrieval changes actually matter to the final answer.**
