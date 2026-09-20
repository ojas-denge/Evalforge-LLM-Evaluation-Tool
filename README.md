# EvalForge

### LLM Evaluation & Observability Platform

EvalForge is an engineering-focused platform for evaluating, debugging, and observing Retrieval-Augmented Generation (RAG) and LLM applications.

The project was built around a simple principle:

> **A system that produces a good answer once is not necessarily a reliable LLM system.**

EvalForge therefore evaluates the layers around generation—not only the final answer—including retrieval quality, groundedness, structured-output validity, latency, token usage, cost, reliability, and regression behavior.

---

## Why EvalForge?

RAG applications are often demonstrated with a few manually selected questions.

That makes it difficult to answer harder engineering questions:

- Did retrieval actually improve?
- Did better retrieval improve the final answer?
- Which evidence was responsible for a failure?
- Did a change introduce a regression elsewhere?
- How much latency or cost did the change add?
- Can a long-running evaluation recover after interruption?
- Can the same system be observed and compared across configurations?

EvalForge was built to make those questions measurable.

---

## Architecture

```text
                           ┌─────────────────────┐
                           │       Client        │
                           └──────────┬──────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │      FastAPI        │
                           │        API          │
                           └──────────┬──────────┘
                                      │
                         ┌────────────┴────────────┐
                         │                         │
                         ▼                         ▼
                 ┌───────────────┐       ┌────────────────┐
                 │   Retrieval   │       │   Evaluation   │
                 │     Layer     │       │     Runner     │
                 └───────┬───────┘       └───────┬────────┘
                         │                       │
                 ┌───────┴────────┐              │
                 ▼                ▼              │
             ChromaDB       SentenceTransformers │
                 │                │               │
                 └───────┬────────┘               │
                         ▼                        │
                  Retrieved Evidence              │
                         │                        │
                         ▼                        │
                  ┌───────────────┐              │
                  │   Generation  │◄─────────────┘
                  │     Layer     │
                  └───────┬───────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Structured      │
                 │ Validation      │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Answer / Judge  │
                 │    Evaluation   │
                 └────────┬────────┘
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
            PostgreSQL          Langfuse
                 │                 │
                 └────────┬────────┘
                          ▼
                 Comparison /
                 Regression / CI
```

### Core layers

| Layer | Responsibility |
|---|---|
| API | FastAPI endpoints and request lifecycle |
| Retrieval | Dense retrieval, candidate selection, optional reranking |
| Generation | Provider-independent LLM generation interface |
| Validation | Structured-output and application-level validation |
| Evaluation | Dataset execution, answer judging, metrics, comparison |
| Reliability | Retries, checkpoint/recovery, failure telemetry |
| Observability | Traces, latency, tokens, cost, correlation metadata |
| Persistence | PostgreSQL evaluation records and ChromaDB retrieval index |
| CI | Automated tests and evaluation/regression infrastructure |

---

## What the project demonstrates

### Evaluation-first RAG

A fixed evaluation dataset is used to measure retrieval and answer behavior instead of relying on a handful of manually selected examples.

### Retrieval experimentation

EvalForge evaluated:

- Dense retrieval
- BM25
- Hybrid retrieval
- CrossEncoder reranking
- Candidate-K behavior
- Chunk-size / overlap variations

### End-to-end diagnosis

Retrieval metrics are kept separate from answer quality so that the project can investigate an important distinction:

> **Improving retrieval ranking does not automatically mean improving generated answers.**

### LLM evaluation

The evaluation pipeline records per-case evidence and evaluates generated responses for correctness and groundedness.

### Reliability

The evaluation runner supports:

- checkpointing
- resumable execution
- bounded retries
- rate-limit handling
- failure telemetry
- batch-oriented workload strategies

### Observability

Telemetry is correlated across evaluation runs, cases, and requests. The system records information such as:

- provider/model
- latency
- token usage
- estimated cost
- retrieval evidence
- validation failures
- generation failures
- retry behavior

### Production hardening

The containerized application includes:

- non-root execution
- embedding model baked into the image
- no runtime dependency on Hugging Face model download
- separate production Compose networking
- PostgreSQL kept off the host network in production mode
- database-backed readiness checking

---

# Experimental Results

The project deliberately preserved both positive and negative findings.

## Retrieval matrix

| Configuration | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | MRR | Approx. latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.8667 | 0.9733 | 0.9867 | 0.7889 | 0.9489 | 0.9800 | 0.9189 | ~21 ms |
| BM25 | 0.8400 | 0.9600 | 0.9867 | 0.7778 | 0.9222 | 0.9600 | 0.9022 | ~1.8 ms |
| Hybrid | 0.8800 | 0.9867 | 1.0000 | 0.8089 | 0.9600 | 0.9889 | 0.9322 | ~23 ms |
| Dense + CrossEncoder | **0.9333** | **1.0000** | **1.0000** | **0.8556** | **0.9733** | **0.9956** | **0.9644** | ~351 ms |
| Hybrid + CrossEncoder | 0.9333 | 1.0000 | 1.0000 | 0.8556 | 0.9733 | 0.9956 | 0.9644 | ~478 ms |

The result was not treated as “CrossEncoder wins, therefore use it everywhere.”

The additional latency and the downstream effect on answer generation were investigated separately.

---

## G1 — 75-case end-to-end experiment

The 75-case experiment compared Dense retrieval against CrossEncoder reranking.

### Dense

- Correct: **75/75**
- Grounded: **75/75**

### Dense + CrossEncoder

- Correct: **73/75**
- Grounded: **75/75**
- Two answer-level regressions were observed.

This became an important engineering finding:

> **A retrieval configuration can improve retrieval metrics while still producing worse end-to-end answers on individual cases.**

---

## G2 — Context ordering investigation

G2 investigated whether the ordering of retrieved context could influence generation.

Two representative cases exposed different failure modes:

- `cost_005`: oracle ordering omitted required synthesis context.
- `golden_004`: the expected document was absent, but secondary evidence still supported a useful answer.

The experiment showed why retrieval correctness, context composition, and answer correctness need to be evaluated separately.

---

## G3C v2 — Targeted end-to-end diagnostic

An initial diagnostic was found to have an experimental-design problem: retrieval metadata was available, but the experiment did not faithfully represent the actual evidence passed to generation.

The experiment was corrected and rerun.

### Corrected 8-case diagnostic

| Configuration | Correct + Grounded |
|---|---:|
| Dense | **7/8** |
| Dense + CrossEncoder | **8/8** |

Outcome:

- 1 improvement
- 7 unchanged
- 0 regressions

The corrected experiment provided evidence that a retrieval change can propagate into answer quality when the retrieved evidence is actually different in a meaningful way.

---

# Engineering Lessons

## 1. Retrieval metrics are not answer metrics

Hit@K, Recall@K, and MRR measure retrieval behavior.

They do not directly measure whether an LLM will produce the desired answer from the retrieved context.

EvalForge therefore keeps retrieval and answer evaluation separate.

## 2. Better ranking can have a cost

CrossEncoder reranking substantially improved retrieval metrics, but it added significant retrieval latency and produced two answer regressions in the 75-case end-to-end experiment.

This is why the project does not reduce system quality to one aggregate score.

## 3. Experiments can be invalid

The first G3C diagnostic did not faithfully model the evidence path.

Rather than treating the output as a result, the experiment was invalidated, corrected, and rerun.

That is intentional engineering behavior.

## 4. Failures are first-class artifacts

Failures are retained because they provide information about:

- assumptions
- system boundaries
- retrieval behavior
- evaluation design
- production constraints
- observability gaps

---

# Evaluation Pipeline

A typical evaluation run follows:

```text
Dataset
   │
   ▼
Evaluation Runner
   │
   ├── checkpoint / resume
   │
   ▼
Application Request
   │
   ├── retrieval
   ├── generation
   └── validation
   │
   ▼
Per-case Result
   │
   ├── retrieval metrics
   ├── answer quality
   ├── groundedness
   ├── latency
   ├── tokens
   ├── cost
   └── failure telemetry
   │
   ▼
Aggregate Evaluation
   │
   ├── comparison
   ├── regression analysis
   └── CI evaluation infrastructure
```

The system preserves per-case evidence so aggregate failures can be investigated instead of disappearing into a single score.

---

# Observability

EvalForge uses Langfuse-compatible tracing and application telemetry.

Trace metadata can correlate:

```text
evaluation_run_id
       │
       └── case_id
               │
               └── request_id
                       │
                       ├── retrieval
                       ├── generation
                       ├── validation
                       └── failure telemetry
```

The observability layer is designed to answer questions such as:

- Which evaluation case caused the regression?
- Which documents were retrieved?
- Which model/provider handled the request?
- How long did generation take?
- How many tokens were consumed?
- Did a retry occur?
- Did the request fail before or after generation?

Content capture is configurable rather than assumed to be always-on.

---

# Production Hardening

The final application container was hardened beyond a development-only RAG demo.

### Container

- Runs as a dedicated non-root user.
- Embedding model is baked into the image.
- Startup does not require downloading the embedding model.
- Writable application paths are explicitly provisioned where required.

### Networking

Development Compose exposes PostgreSQL for local integration testing.

Production Compose removes the PostgreSQL host port while keeping the API-to-database network path available internally.

### Health

Two health semantics are exposed:

```text
GET /health
```

Liveness/service health.

```text
GET /health/ready
```

Readiness including a database connectivity check.

A database failure causes readiness to return HTTP 503 instead of reporting the service as ready.

---

# Tech Stack

| Area | Technology |
|---|---|
| Language | Python |
| API | FastAPI |
| Validation | Pydantic |
| Database | PostgreSQL |
| Vector store | ChromaDB |
| Embeddings | SentenceTransformers / BAAI bge-small-en-v1.5 |
| Retrieval | Dense, BM25, hybrid, CrossEncoder |
| LLM interface | OpenAI-compatible provider abstraction |
| Observability | Langfuse |
| Containerization | Docker / Docker Compose |
| Testing | pytest / HTTPX |
| CI | GitHub Actions |

---

# Repository Structure

```text
evalforge/
├── app/
│   ├── api/
│   ├── core/
│   ├── db/
│   ├── evaluation/
│   ├── generation/
│   ├── models/
│   ├── observability/
│   └── retrieval/
│
├── data/
│   ├── documents/
│   └── evaluation_cases.json
│
├── evaluation_baselines/
│
├── experiments/
│
├── logs/
│
├── scripts/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── docker-compose.prod.yml
├── pyproject.toml
└── README.md
```

---

# Running EvalForge

## Local Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[test]"
```

Configure the required environment variables in `.env`.

Do not commit `.env`.

Start the application:

```powershell
uvicorn app.main:app --reload
```

API documentation:

```text
http://localhost:8000/docs
```

Health:

```text
http://localhost:8000/health
```

Readiness:

```text
http://localhost:8000/health/ready
```

---

## Docker — development

```powershell
docker compose up --build
```

This configuration exposes PostgreSQL for local integration testing.

---

## Docker — production-style local stack

```powershell
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

In this configuration PostgreSQL is reachable by the API through the Compose network but is not published to the host.

---

# Testing

Run the full test suite:

```powershell
pytest -q
```

The final local validation reached:

```text
165 passed
1 warning
```

The remaining warning is a dependency deprecation warning originating from the Starlette/AnyIO stack rather than an EvalForge test failure.

---

# Current Scope and Limitations

EvalForge is an engineering evaluation platform, not a claim of production-scale infrastructure.

Current limitations include:

- The benchmark is based on a project-specific evaluation corpus.
- Retrieval experiments do not establish universal superiority across datasets.
- LLM judge outputs are themselves model-dependent and require reliability checks.
- Token/cost values depend on the configured provider and pricing.
- The current application is a local/containerized deployment rather than a horizontally scaled service.
- The production Compose configuration is a hardening step, not a complete production deployment platform.
- The experiments demonstrate measured behavior on the available benchmark; they should not be generalized beyond that evidence.

---

# Evaluation Workflow

The evaluation runner executes from the host Python environment and connects to PostgreSQL through `localhost:5432`.

Start the development Compose stack before running host-side evaluations:

```powershell
docker compose up -d
```

Initialize the database if required:

```powershell
python -c "from app.db.session import init_db; init_db()"
```

Run an evaluation:

```powershell
python .\\scripts\\run_evaluation.py
```

Resume an interrupted evaluation:

```powershell
python .\\scripts\\run_evaluation.py --resume <RUN_ID>
```

The production Compose configuration intentionally does not expose PostgreSQL on the host. Therefore, host-side evaluation should be run against the development Compose configuration.

## Documentation

The repository's detailed engineering record is maintained separately from this portfolio-facing README.

See:

```text
TECHNICAL_DOCUMENTATION.md
```

The technical documentation contains the deeper implementation history, experiment rationale, failure analysis, production hardening details, reproducibility information, and engineering decisions.

---

# Project Status

- **Engineering implementation:** Complete
- **Experimentation track:** Closed
- **Production-hardening pass:** Complete
- **Technical documentation:** Complete
- **Portfolio presentation:** Finalized
- **Final repository audit:** Pending

---

## Engineering Philosophy

EvalForge was developed around:

```text
Implement
   ↓
Test
   ↓
Measure
   ↓
Investigate
   ↓
Document
   ↓
Commit
```

The objective was not to maximize the number of features.

The objective was to build a system where engineering decisions can be explained with:

**evidence → experiment → failure analysis → decision.**

---

## Author

**Ojas Denge**

AI Engineer · LLM Systems · Agentic AI

Focus areas:

- LLM evaluation
- RAG systems
- AI/LLM observability
- agentic workflows
- retrieval systems
- production-oriented AI engineering
