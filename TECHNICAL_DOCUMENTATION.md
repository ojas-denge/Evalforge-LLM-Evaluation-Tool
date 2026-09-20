# EvalForge — Final Technical Documentation

> **LLM Evaluation & Observability Platform**
>
> A production-oriented evaluation system for retrieval-augmented generation and LLM applications, built around measurable engineering decisions, failure diagnosis, observability, and regression protection.

**Documentation status:** Final technical documentation draft  
**Project version:** `0.1.0`  
**Engineering checkpoint:** September 20, 2026

---

# 1. Executive Summary

EvalForge was built to answer a practical engineering question:

> **How do we know whether an LLM application is actually getting better, and how do we know when it gets worse?**

The project evolved from a baseline RAG application into an evaluation and observability platform with:

- a fixed golden evaluation dataset,
- measurable retrieval quality,
- multiple retrieval strategies,
- CrossEncoder reranking,
- candidate-K investigation,
- generation abstraction,
- provider/model provenance,
- structured-output parsing and validation,
- LLM-as-a-judge evaluation,
- judge reliability and bounded recovery,
- checkpointed evaluation runs,
- retry and rate-limit handling,
- Langfuse observability,
- PostgreSQL persistence,
- regression/comparison infrastructure,
- CI evaluation gates,
- Docker production hardening,
- non-root execution,
- a baked-in embedding model,
- liveness and database readiness checks.

The central engineering principle remained consistent throughout:

```text
Hypothesis
   ↓
Baseline
   ↓
Measurement
   ↓
Failure
   ↓
Diagnosis
   ↓
Controlled experiment
   ↓
Implementation correction
   ↓
Re-measurement
   ↓
Engineering decision
   ↓
Tests / regression protection
   ↓
Observability
```

EvalForge is therefore not primarily a collection of AI technologies. It is a collection of **measured engineering decisions**.

---

# 2. Problem Definition

LLM applications are difficult to evaluate because a single successful response does not establish system quality.

A RAG system can:

- retrieve the wrong evidence,
- retrieve the right evidence at the wrong rank,
- retrieve enough evidence but fail to synthesize it,
- produce a grounded but incomplete answer,
- produce a correct answer using secondary evidence,
- return valid JSON with the wrong application-level schema,
- return HTTP 200 with missing or truncated content,
- silently use a different model than the configured model,
- become slower or more expensive while quality improves,
- pass aggregate metrics while hiding important per-case regressions.

EvalForge therefore treats evaluation as a multi-dimensional engineering problem.

The system separates:

```text
Retrieval quality
        +
Answer quality
        +
Groundedness
        +
Structured-output validity
        +
Latency
        +
Token usage
        +
Cost
        +
Reliability
        +
Provenance
```

No single metric is treated as sufficient evidence of improvement.

---

# 3. Design Principles

## 3.1 Measure before optimizing

The baseline was measured before introducing retrieval optimizations.

## 3.2 Preserve per-case evidence

Aggregate metrics are useful for detecting change, but individual cases are required to explain why the change occurred.

## 3.3 Separate pipeline stages

Retrieval failures, generation failures, validation failures, judge failures, and provider failures are different events and should remain distinguishable.

## 3.4 Treat evaluators as systems that require evaluation

The LLM judge is itself probabilistic and can fail through malformed output, missing content, schema violations, or provider errors.

## 3.5 Preserve actual provider provenance

The configured model is not always the model that actually produced a response, especially with dynamic routing.

## 3.6 Validate at the application boundary

Provider-side structured output is useful but is not considered sufficient application-level validation.

## 3.7 Make experiments reproducible

Experiments use fixed datasets, explicit configurations, and preserved artifacts.

## 3.8 Keep experiments isolated from production behavior

Experimental artifacts live under `experiments/` and do not silently alter the production evaluation path.

## 3.9 Prefer measured trade-offs over isolated maxima

A higher retrieval score is not automatically better if it introduces unnecessary latency without downstream benefit.

---

# 4. System Architecture

The current architecture can be represented as:

```text
                         EvalForge
                            │
             ┌──────────────┴──────────────┐
             │                             │
          API Layer                 Evaluation Layer
             │                             │
             ▼                             ▼
        FastAPI app                 Dataset / Runner
             │                             │
             └──────────────┬──────────────┘
                            │
                            ▼
                      Retrieval Layer
                            │
                ┌───────────┴───────────┐
                │                       │
             ChromaDB             SentenceTransformers
                │
                ▼
          Retrieved Evidence
                │
                ▼
          Generation Layer
                │
        ┌───────┴────────┐
        │                │
  Deterministic     OpenAI-compatible
    generator          generator
        │                │
        └───────┬────────┘
                ▼
          GenerationResult
                │
                ▼
        Structured Validation
                │
                ▼
          Answer Judge
                │
                ▼
         EvaluationResult
                │
        ┌───────┴────────┐
        │                │
   PostgreSQL       Observability
        │             / Langfuse
        │                │
        └───────┬────────┘
                ▼
       Comparison / Regression
                │
                ▼
             CI Gate
```

---

# 5. Technology Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Runtime | Python 3.12 |
| Validation | Pydantic |
| Database | PostgreSQL 17 |
| ORM / DB access | SQLAlchemy + psycopg |
| Vector retrieval | ChromaDB |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Embedding dimension | 384 |
| Generation abstraction | Provider-independent `Generator` interface |
| Real provider adapter | OpenAI-compatible chat-completions |
| Structured output | JSON Schema / JSON Object modes |
| Observability | Langfuse |
| Containerization | Docker / Docker Compose |
| Testing | pytest |
| CI | GitHub Actions |
| Reranking experiment | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

---

# 6. Repository Architecture

The major application areas are:

```text
app/
├── api / main application
├── core/
├── db/
├── evaluation/
│   ├── answer_judge.py
│   ├── candidate_diagnostics.py
│   ├── ci_gate.py
│   ├── comparison.py
│   ├── conflict_detector.py
│   ├── dataset.py
│   ├── diagnostics.py
│   ├── judge_reliability.py
│   ├── judge_result.py
│   ├── metrics.py
│   ├── regression.py
│   ├── reporting.py
│   ├── runner.py
│   ├── structured_output.py
│   ├── triage.py
│   └── trust.py
├── generation/
│   ├── base.py
│   ├── deterministic.py
│   ├── factory.py
│   ├── observation.py
│   ├── openai_compatible.py
│   ├── pricing.py
│   ├── pricing_registry.py
│   ├── structured.py
│   └── usage.py
├── models/
├── observability/
└── retrieval/

experiments/
├── retrieval/
├── g1/
├── g2/
├── g3/
├── legacy/
└── results/

evaluation_baselines/
scripts/
tests/
.github/workflows/
```

The separation is intentional:

- production code remains under `app/`,
- experiments remain under `experiments/`,
- evaluation baselines remain explicit,
- tests protect application contracts,
- CI invokes the evaluation path independently.

---

# 7. Retrieval System

## 7.1 Baseline Retriever

The baseline dense retriever uses:

- `BAAI/bge-small-en-v1.5`
- 384-dimensional embeddings
- ChromaDB
- top-k retrieval
- current baseline `top_k = 5`

The corpus used in the experimentation program contained:

- 16 source documents,
- 17 baseline chunks,
- 75 evaluation cases.

## 7.2 Retrieval Metrics

EvalForge records:

- Hit@1
- Hit@3
- Hit@5
- Recall@1
- Recall@3
- Recall@5
- MRR
- retrieval latency

The metrics answer different questions.

For example:

- Hit@K asks whether expected evidence appears in the first K results.
- Recall@K measures how much expected evidence is retrieved.
- MRR captures the rank of the first relevant result.

This distinction became important during multi-document experiments.

---

# 8. Retrieval Strategy Experiments

## 8.1 Strategy Matrix

The 75-case benchmark compared:

| Strategy | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | MRR | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.8667 | 0.9733 | 0.9867 | 0.7889 | 0.9489 | 0.9800 | 0.9189 | ~21 ms |
| BM25 | 0.8400 | 0.9600 | 0.9867 | 0.7778 | 0.9222 | 0.9600 | 0.9022 | ~1.8 ms |
| Hybrid | 0.8800 | 0.9867 | 1.0000 | 0.8089 | 0.9600 | 0.9889 | 0.9322 | ~23 ms |
| Dense + CrossEncoder | 0.9333 | 1.0000 | 1.0000 | 0.8556 | 0.9733 | 0.9956 | 0.9644 | ~351 ms |
| Hybrid + CrossEncoder | 0.9333 | 1.0000 | 1.0000 | 0.8556 | 0.9667 | 0.9956 | 0.9644 | ~478 ms |

The experimental conclusion was benchmark-specific:

> Dense + CrossEncoder achieved the same primary quality figures as Hybrid + CrossEncoder with lower measured latency on this benchmark.

This does **not** establish universal superiority.

---

# 9. CrossEncoder Forensics

The CrossEncoder changed ranking frequently without necessarily changing the evidence set.

Across 75 cases:

- 52 neutral reorderings,
- 9 ranking improvements,
- 13 unchanged cases,
- 1 ranking regression.

Measured dense → CrossEncoder change:

| Metric | Dense | CrossEncoder | Delta |
|---|---:|---:|---:|
| Hit@1 | 0.8667 | 0.9467 | +0.0800 |
| Hit@3 | 0.9733 | 0.9867 | +0.0133 |
| Hit@5 | 0.9867 | 0.9867 | 0 |
| Recall@1 | 0.7889 | 0.8689 | +0.0800 |
| Recall@3 | 0.9489 | 0.9644 | +0.0156 |
| Recall@5 | 0.9800 | 0.9800 | 0 |
| MRR | 0.9189 | 0.9644 | +0.0456 |

A key forensic case was `routing_006`.

Dense ranked `model-routing.md` first and achieved MRR 1.0.

CrossEncoder moved `deployment-policy.md` above it, producing MRR 0.5 while keeping the required evidence available.

This exposed a multi-evidence ranking weakness:

> A pairwise relevance improvement can conflict with the requirements of multi-document synthesis.

---

# 10. Candidate-K Investigation

Candidate pools of 5, 10, 15, and 17 were compared.

| Candidate K | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | MRR | Approx. latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.9467 | 0.9867 | 0.9867 | 0.8689 | 0.9644 | 0.9800 | 0.9644 | ~159 ms |
| 10 | 0.9333 | 1.0000 | 1.0000 | 0.8556 | 0.9733 | 0.9956 | 0.9644 | ~386 ms |
| 15 | same as K=10 | same | same | same | same | same | same | ~569 ms |
| 17 | same as K=10 | same | same | same | same | same | same | ~650 ms |

Only two cases gained expected evidence from the larger candidate pool.

Quality plateaued beyond approximately K=10.

A real implementation bug was also found: an explicit `candidate_k` had previously been silently expanded by another retrieval parameter. The implementation was corrected so that explicit candidate K is authoritative unless expansion is required to satisfy the requested output size.

The selected operating point for the benchmark was K=5 because it retained the same MRR while reducing latency.

---

# 11. Chunking Ablation

The baseline configuration was:

```text
chunk_size = 120
chunk_overlap = 30
```

Chunk-size comparison:

| Configuration | MRR | Hit@1 | Recall@5 |
|---|---:|---:|---:|
| 120 / 30 | 0.9189 | 0.8667 | 0.9800 |
| 60 / 15 | 0.9071 | 0.8533 | 0.9689 |
| 240 / 60 | 0.9256 | 0.8800 | 0.9800 |

The 240/60 configuration showed a small improvement, but the evidence did not justify changing the operating configuration during the sprint.

Overlap sweeps also showed similar aggregate quality.

Conclusion:

> Chunking affects ranking, but the measured evidence was insufficient to justify changing the baseline configuration.

---

# 12. End-to-End RAG Evaluation

The project deliberately moved beyond retrieval-only metrics.

The end-to-end evaluation tracks:

- answer correctness,
- groundedness,
- topic coverage,
- missing topics,
- unsupported claims,
- generation latency,
- token usage,
- estimated cost,
- structured-output validity.

This distinction is central to EvalForge.

A retrieval improvement is not automatically a RAG improvement.

---

# 13. G1 — Retrieval → Generation Propagation

Two conditions were compared:

1. Dense → top-5 → LLM
2. Dense → CrossEncoder → top-5 → LLM

Using the 75-case benchmark:

### Dense

- 75/75 correct
- 75/75 grounded
- 0 unsupported-claim cases
- mean topic coverage approximately 97.87%

### Dense + CrossEncoder

- 73/75 correct
- 75/75 grounded
- 0 unsupported-claim cases
- mean topic coverage approximately 97.87%

Propagation:

- 0 answer improvements,
- 73 unchanged,
- 2 regressions.

The important result was not that CrossEncoder was “bad.”

The important result was:

> Retrieval-ranking improvement did not automatically propagate into end-to-end answer improvement.

---

# 14. G2 — Context Ordering

G2 isolated ordering from evidence availability.

Conditions included:

- Dense,
- Dense + CrossEncoder,
- expected-first ordering,
- oracle ordering.

The interventions reordered evidence rather than adding new evidence.

Two important diagnostic cases:

## `cost_005`

The evidence supported the required answer, but an oracle ordering caused the generated answer to omit an important synthesis relationship.

## `golden_004`

The expected document was absent, but the system still produced a correct answer from secondary evidence under one condition.

The experiment established:

> Evidence availability, evidence ordering, and semantic synthesis are separate variables.

---

# 15. G3 — Failure Mining and Targeted Diagnosis

The original 75-case benchmark was highly answerable at the generation layer.

G3 therefore moved from broad benchmark measurement toward targeted diagnostics.

## G3A

Failure mining identified retrieval-insensitive cases and a small set of potentially retrieval-sensitive cases.

## G3B

A 16-case screening set was used to find cases with meaningful ranking changes.

The screening showed:

- document-set changes: 0/16,
- order changes: 16/16,
- expected-rank changes: 13/16.

A 10-case end-to-end diagnostic run produced:

- Dense: 10/10 correct and grounded,
- CrossEncoder: 10/10 correct and grounded,
- 0 improvements,
- 10 unchanged,
- 0 regressions.

## G3C

The final diagnostic set was reduced to 8 targeted cases.

The corrected G3C v2 experiment passed the actual retrieved document text into generation and converted the same evidence into the judge representation.

Results:

| Condition | Correct | Grounded |
|---|---:|---:|
| Dense | 7/8 | 7/8 |
| Dense + CrossEncoder | 8/8 | 8/8 |

Propagation:

- 1 answer improvement,
- 7 unchanged,
- 0 regressions.

`cross_001` finally demonstrated a direct retrieval → answer propagation case.

The Dense condition had sufficient evidence available but produced an incomplete answer. The CrossEncoder ordering produced the expected answer.

This established:

> Retrieval ranking can affect final answer quality even when the evidence set does not change.

The G3C experiment remains a targeted 8-case diagnostic, not a replacement for the 75-case benchmark.

---

# 16. Generation Architecture

Generation is provider-independent.

The core contract is:

```text
GenerationRequest
        ↓
Generator
        ↓
GenerationResult
```

A request can contain:

- question,
- retrieved context,
- model,
- temperature,
- max tokens,
- system prompt,
- response schema,
- metadata.

A result preserves:

- answer,
- requested model,
- provider,
- token usage,
- estimated cost,
- latency,
- finish reason,
- structured output,
- structured-output status,
- observation,
- metadata.

Two important implementations exist:

```text
Generator
├── deterministic generator
└── OpenAI-compatible generator
```

The deterministic generator exists primarily for reliable application tests.

The provider adapter is used for real model experiments.

---

# 17. Provider Provenance

The real provider adapter preserves both:

```text
requested model
actual model
actual provider
```

This matters because dynamic routing can cause the configured model identifier to differ from the model that actually produced the response.

Provider observations also preserve:

- HTTP status,
- finish reason,
- raw response,
- reasoning fields when supplied,
- usage,
- latency,
- response format,
- content state.

This makes the evaluation evidence auditable.

---

# 18. Structured Output

EvalForge distinguishes:

```text
NOT_REQUESTED
PARSED
MISSING_CONTENT
PARSE_FAILED
```

Provider-side structured output is parsed first.

Application-level validation then verifies that the parsed object satisfies the expected Pydantic contract.

This distinction matters because:

> Valid JSON is not equivalent to a valid application response.

The project specifically tested cases where syntactically valid JSON violated the expected field contract.

Application-side schema validation therefore remains mandatory.

---

# 19. LLM-as-a-Judge

The answer judge evaluates:

- answer correctness,
- groundedness,
- expected topic coverage,
- missing topics,
- unsupported claims,
- reasoning.

The judge uses structured output and a strict application-side validator.

The system supports both:

```text
single-case judge
```

and:

```text
batch judge
```

Batch judging validates that:

- the provider returns structured output,
- the verdict array exists,
- the verdict count matches the case count,
- each verdict validates against the expected model,
- verdicts are mapped back to cases in input order.

This prevents silent case/verdict misalignment.

---

# 20. Judge Reliability

The evaluator treats the judge itself as a component that can fail.

Observed failure classes include:

- missing content,
- malformed structured output,
- schema validation failure,
- provider/system error.

Semantic structured-output failures can be retried with a bounded retry policy.

Transport/provider failures are kept distinct from semantic repair.

This distinction is important:

```text
semantic repair
      ≠
provider transport retry
```

Judge metadata preserves:

- attempt count,
- initial failure,
- failure type,
- whether recovery occurred,
- model/provider provenance,
- usage,
- cost.

---

# 21. Evaluation Runner

The evaluation runner orchestrates:

```text
Dataset
   ↓
Retrieval
   ↓
Generation
   ↓
Per-case metrics
   ↓
Checkpoint
   ↓
Judge
   ↓
Checkpoint
   ↓
Aggregation
   ↓
EvaluationRun
```

The runner preserves per-case results rather than storing only aggregate metrics.

This allows later comparison and failure triage.

---

# 22. Checkpoint and Recovery Design

Long-running evaluation runs are vulnerable to:

- rate limits,
- network failures,
- provider errors,
- process interruption.

EvalForge therefore stores checkpoints containing:

- run ID,
- phase,
- completed case IDs,
- serialized results,
- retry state.

Generation and judge phases maintain separate completion state.

This allows an interrupted run to resume rather than restarting from zero.

Retryable conditions include:

- HTTP 429,
- rate-limit indicators,
- quota/resource exhaustion,
- transient read timeouts.

Retry behavior is bounded.

The project deliberately avoids unbounded retry loops.

---

# 23. Batch Evaluation and Provider Limits

During the evaluation program, individual external generation requests encountered provider rate limits.

Batching reduced request pressure substantially.

Measured batch experiments showed:

### 5 cases/request

- 75/75 processed,
- 15 requests,
- 0 integrity failures,
- 80% request reduction,
- approximately 36.88 seconds wall time.

### 15 cases/request

- 75/75 processed,
- 5 requests,
- 0 integrity failures,
- 93.3% request reduction,
- approximately 13.51 seconds wall time.

The batching experiment preserved exact case count and ordering.

The experimentation program treated batching as an evaluation reliability and provider-capacity problem, not merely a performance trick.

---

# 24. Observability

Langfuse instrumentation was hardened so that a request can be correlated across:

```text
evaluation_run_id
case_id
request_id
```

Telemetry captures or associates:

- model,
- provider,
- retrieval metadata,
- generation metadata,
- token usage,
- latency,
- cost,
- errors,
- retrieval evidence metadata.

Content capture is controlled and can be redacted.

The goal is not merely to collect traces.

The goal is to make failures explainable.

---

# 25. Production API

The FastAPI application exposes:

```text
GET /health
GET /health/ready
POST /query
GET /evaluations
GET /evaluations/{run_id}
```

The `/query` path performs the real retrieval → generation flow and returns:

- answer,
- citations,
- usage,
- cost,
- latency,
- request metadata.

The system was validated with a real production-style query after container hardening.

---

# 26. Database and Persistence

PostgreSQL is used for durable evaluation persistence.

The production container reaches PostgreSQL through the Compose network.

Development/test environments can expose PostgreSQL to the host for integration tests.

Production Compose removes the host PostgreSQL port exposure.

This preserves:

```text
API → internal database network
```

without unnecessarily exposing the database externally.

---

# 27. Production Container Hardening

The final Docker image includes:

- Python 3.12 slim,
- application dependencies,
- application source,
- evaluation data,
- embedding model.

The embedding model:

```text
BAAI/bge-small-en-v1.5
```

is baked into the image at:

```text
/opt/models/bge-small-en-v1.5
```

This avoids runtime Hugging Face downloads.

The container runs as:

```text
uid=10001(evalforge)
```

rather than root.

A writable checkpoint directory is explicitly provisioned for the non-root user.

---

# 28. Production Networking

The development Compose configuration exposes PostgreSQL for local integration testing.

The production override removes the PostgreSQL host port.

Validated production behavior:

```text
localhost:5432
    → not externally reachable

API container
    ↓
postgres:5432
    → reachable internally
```

The API therefore retains database connectivity without publishing PostgreSQL to the host.

---

# 29. Health and Readiness

Two health concepts are intentionally separated.

## Liveness

```text
GET /health
```

Confirms the API process is alive.

## Readiness

```text
GET /health/ready
```

Performs:

```sql
SELECT 1
```

against PostgreSQL.

Successful response:

```json
{
  "status": "ready",
  "service": "evalforge",
  "version": "0.1.0"
}
```

Database failure produces HTTP 503.

Both paths were covered by tests, including the database-unavailable case.

---

# 30. Testing

The project maintains unit and integration tests across:

```text
tests/
├── api/
├── db/
├── evaluation/
├── generation/
├── retrieval/
└── observability/
```

Generation tests explicitly protect:

- response parsing,
- structured output,
- malformed JSON handling,
- schema boundary behavior,
- null content,
- provider provenance,
- reasoning metadata,
- usage,
- latency,
- raw provider response preservation.

The evaluation tests protect:

- metrics,
- dataset handling,
- judge behavior,
- judge reliability,
- regression logic,
- CI gate behavior,
- checkpoint/recovery behavior.

The latest full local suite after production-readiness work:

```text
165 passed
1 warning
```

The warning is an upstream Starlette/AnyIO deprecation warning and is not an EvalForge test failure.

---

# 31. CI / Regression Architecture

The repository contains explicit components for:

```text
comparison
regression
CI gate
reporting
triage
trust
```

The intended evaluation flow is:

```text
Fixed baseline
      ↓
Candidate evaluation
      ↓
Metric comparison
      ↓
Per-case comparison
      ↓
Regression analysis
      ↓
Quality gate
      ↓
CI PASS / FAIL
```

The baseline is stored explicitly under:

```text
evaluation_baselines/
```

This is important because regression protection should compare a candidate against a known reference rather than rely on an informal human comparison.

---

# 32. Experimental Artifact Organization

Experimental evidence is kept under:

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

Earlier invalid or superseded experiments are preserved where useful for provenance, but are not treated as final evidence.

In particular, G3C v1 is retained as experimental history but G3C v2 is the valid final result because it passes the actual retrieved text into generation and the correct evidence representation into judging.

---

# 33. Failure Analysis

Important failures discovered during development include:

## Retrieval

An implementation bug caused explicit candidate K to be overridden.

**Resolution:** make explicit candidate K authoritative.

## Generation

A provider returned HTTP success with:

```text
finish_reason = length
content = incomplete
```

**Resolution:** preserve incomplete content state and finish reason instead of treating HTTP 200 as application success.

## Structured output

Valid JSON violated the application-level field contract.

**Resolution:** strict application-side validation.

## Provider routing

A dynamic free-model identifier routed to different actual models/providers.

**Resolution:** record actual model/provider provenance.

## Judge

The judge produced missing or invalid structured output.

**Resolution:** bounded semantic retry and explicit judge failure states.

## Provider quota

A long-running external evaluation exceeded available provider request capacity.

**Resolution:** checkpointing, bounded retry, and batch-oriented evaluation experiments.

## Production

Runtime embedding downloads created an undesirable network dependency.

**Resolution:** bake the embedding model into the image.

## Production

The container initially ran as root.

**Resolution:** dedicated non-root runtime user.

## Networking

PostgreSQL was unnecessarily exposed in the production stack.

**Resolution:** production Compose override removes the host port.

## Readiness

Liveness alone could report a healthy API while the database was unavailable.

**Resolution:** database-backed readiness endpoint.

---

# 34. What the Experiments Actually Proved

The experimentation program established several important conclusions.

### Retrieval

1. Dense retrieval is a strong baseline for the current corpus.
2. BM25 is substantially faster but somewhat weaker on the benchmark.
3. Hybrid retrieval improves retrieval metrics over dense retrieval.
4. CrossEncoder reranking produces the strongest measured ranking quality in this benchmark.
5. Reranking frequently changes ordering without changing evidence availability.
6. Larger candidate pools have diminishing returns.
7. Candidate K above approximately 10 showed no additional measured quality benefit.
8. Chunking affects retrieval ranking, but the evidence did not justify changing the baseline configuration.

### Generation

9. Better retrieval metrics do not automatically produce better answers.
10. LLMs can compensate for some retrieval failures using secondary evidence.
11. Context ordering can affect generation even when the evidence set is unchanged.
12. Multi-document synthesis is particularly sensitive to ranking.

### Evaluation

13. Retrieval quality and answer quality must be measured separately.
14. A broad benchmark can be too answerable to expose retrieval-to-generation propagation.
15. Targeted diagnostic cases are useful for studying causal propagation.
16. The evaluator itself requires reliability engineering.

### Engineering

17. Provider success must not be equated with application success.
18. Application-level structured validation is necessary.
19. Actual provider provenance should be preserved.
20. Checkpointing is essential for long-running external evaluations.
21. Batch experiments can substantially reduce provider request pressure.
22. Experimental evidence should remain tied to the dataset and conditions under which it was measured.

---

# 35. Final Experimental Status

| Experiment | Status |
|---|---|
| Retrieval Strategy Matrix | Complete |
| CrossEncoder Forensics | Complete |
| Candidate-K | Complete |
| Chunking Ablation | Complete |
| G1 Retrieval → Generation | Complete |
| G2 Context Ordering | Complete |
| G3A Failure Mining | Complete |
| G3B Candidate Screening | Complete |
| G3B End-to-End | Complete |
| G3C Candidate Mining | Complete |
| G3C Retrieval Checkpoint | Complete |
| G3C End-to-End v2 | Complete |

**Experimentation track: CLOSED.**

No G4+ retrieval experiment is required for the current project scope.

---

# 36. Final Project Status

| Workstream | Status |
|---|---|
| Foundation | Complete |
| Baseline RAG | Complete |
| Retrieval evaluation | Complete |
| Retrieval diagnostics | Complete |
| Retrieval strategy comparison | Complete |
| Reranking | Complete |
| Candidate-K investigation | Complete |
| Chunking investigation | Complete |
| Generation abstraction | Complete |
| Provider provenance | Complete |
| Structured outputs | Complete |
| Answer-level evaluation | Complete |
| LLM judge | Complete |
| Judge reliability | Complete |
| Checkpoint/recovery | Complete |
| Observability | Complete |
| PostgreSQL persistence | Complete |
| Regression/comparison infrastructure | Implemented |
| CI evaluation gate infrastructure | Implemented |
| Production container hardening | Complete |
| Non-root execution | Complete |
| Baked embedding model | Complete |
| Production networking hardening | Complete |
| Liveness/readiness | Complete |
| Technical experimentation | Closed |
| Final technical documentation | This document |
| Portfolio presentation | Separate finalization layer |

---

# 37. Known Limitations

The conclusions in this document are bounded by the tested environment.

## Benchmark scope

The retrieval conclusions are based on the current corpus and 75-case benchmark.

They should not be generalized to arbitrary corpora.

## Model/provider variance

External provider behavior can change independently of EvalForge.

Dynamic routing and provider quotas can influence observed evaluation behavior.

## Cost estimation

Estimated cost depends on the configured pricing registry and observed token usage.

It is not equivalent to a provider invoice.

## Judge subjectivity

LLM-as-a-judge is itself a probabilistic evaluation component.

Its verdicts should therefore be interpreted as evaluation evidence rather than unquestionable ground truth.

## Targeted diagnostic sample size

G3C's 8-case result is strong diagnostic evidence for the studied cases but is not a statistically representative replacement for the 75-case benchmark.

## Agent evaluation

The current project establishes the evaluation foundation for agentic applications but does not claim that a complete agent-specific benchmark has been implemented.

---

# 38. Reproducibility

The project is designed around explicit configuration and reproducible evaluation artifacts.

Core local setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[test]"
```

Run tests:

```powershell
pytest -q
```

Development stack:

```powershell
docker compose up -d
```

Production-style stack:

```powershell
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

Production health:

```text
GET /health
GET /health/ready
```

The repository's environment example should be used for configuration without committing real credentials.

---

# 39. Final Engineering Narrative

The strongest way to describe the project is not:

> "I built a RAG application with FastAPI, ChromaDB, Langfuse, and Gemini."

That describes technologies.

The actual engineering narrative is:

```text
A RAG system was built.
        ↓
A fixed benchmark was established.
        ↓
Retrieval was measured before optimization.
        ↓
Multiple retrieval strategies were compared.
        ↓
CrossEncoder reranking improved retrieval ranking.
        ↓
Forensics showed that ranking improvements often
did not change evidence availability.
        ↓
Candidate-K experiments exposed diminishing returns.
        ↓
A retrieval implementation bug was discovered and fixed.
        ↓
Generation was abstracted from the provider.
        ↓
Real provider behavior exposed provenance and
structured-output problems.
        ↓
Application-level validation was added.
        ↓
An LLM judge was built.
        ↓
The judge itself was instrumented and made recoverable.
        ↓
End-to-end experiments showed that retrieval metrics
do not automatically translate into better answers.
        ↓
Targeted G3 experiments isolated context-ordering
and multi-evidence effects.
        ↓
Observability was hardened so failures became explainable.
        ↓
Evaluation became checkpointed and recoverable.
        ↓
Production deployment was hardened.
        ↓
The system became regression-aware and CI-oriented.
```

That is the core technical story of EvalForge.

---

# 40. Final One-Line Finding

> **EvalForge demonstrates that serious LLM engineering requires more than improving model or retrieval metrics: the system must measure retrieval, generation, groundedness, cost, latency, reliability, and provenance together, then preserve enough evidence to explain why a change helped, failed, or appeared to help without actually improving the final application.**

---

# 41. Documentation Boundary

This document is the **technical record** of the project.

It intentionally does not attempt to be:

- a recruiter-oriented README,
- a resume,
- a marketing page,
- a demo script,
- or a personal-branding document.

Those belong to the portfolio-presentation phase after the technical repository is finalized.

The purpose of this document is to preserve:

```text
what was built
+
why it was built
+
what was measured
+
what failed
+
what was changed
+
what the experiments proved
+
what remains bounded or uncertain
```

That separation keeps the engineering record honest and makes the later portfolio presentation substantially easier.
