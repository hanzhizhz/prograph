# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProGraph is a multi-hop question answering system based on **Atomic Propositions** and **Intent-Driven Grouped Beam Search**. The system builds a heterogeneous graph from documents where nodes are atomic propositions and entities, with RST-based rhetorical relationships between propositions.

## Common Commands

### Python Environment

Use the conda environment for this project:
```bash
/home/ubuntu/miniconda3/envs/vllm/bin/python
```

### Testing
```bash
# Run all tests
bash run_tests.sh

# Run specific test
python tests/test_basic.py
python tests/test_timing.py
```

### System Architecture: Two Phases

**Offline Phase** - Graph construction using vLLM offline inference:
- Graph building (`scripts/1-build_proposition_graph.py`)
- Entity linking candidates (`scripts/2a-generate_candidates.py`)
- Entity linking & fusion (`scripts/2b-link_and_fuse.py`)

**Online Phase** - Question answering using API requests:
- Multi-hop QA (`scripts/5-run_multi_hop_qa.py`)

### Building the Knowledge Graph (Offline Pipeline)

**Step 1: Build proposition graph**
```bash
python scripts/1-build_proposition_graph.py \
  --dataset dataset/HotpotQA/full_docs.json \
  --output output/HotpotQA/proposition_graph/raw_graph \
  --config config.yaml
```

**Step 2: Entity linking and graph fusion**

Two options - use two-stage if GPU memory is limited, or integrated if sufficient memory:

**Option A: Two-stage (recommended for limited GPU)**
```bash
# Stage 1: Generate candidate pairs (vector model only)
python scripts/2a-generate_candidates.py \
  --graph output/HotpotQA/proposition_graph/raw_graph.pkl \
  --output output/HotpotQA/temp \
  --config config.yaml

# Stage 2: Link entities and fuse graph (LLM only)
python scripts/2b-link_and_fuse.py \
  --graph output/HotpotQA/proposition_graph/raw_graph.pkl \
  --temp_dir output/HotpotQA/temp \
  --output output/HotpotQA/proposition_graph/linked_graph \
  --config config.yaml
```

**Option B: Integrated (requires more GPU memory)**
```bash
python scripts/2-link_entities_fuse_graph.py \
  --graph output/HotpotQA/proposition_graph/raw_graph.pkl \
  --output output/HotpotQA/proposition_graph/linked_graph \
  --config config.yaml
```

### Online QA

```bash
# Single question
python scripts/5-run_multi_hop_qa.py \
  --question "What is the capital of France?" \
  --graph output/HotpotQA/proposition_graph/linked_graph.pkl \
  --config config.yaml

# Batch processing
python scripts/5-run_multi_hop_qa.py \
  --dataset dataset/HotpotQA/test_data.json \
  --graph output/HotpotQA/proposition_graph/linked_graph.pkl \
  --output output/HotpotQA/result.json \
  --config config.yaml \
  --concurrency 50
```

## High-Level Architecture

### Graph Structure

The graph is a NetworkX DiGraph with three node types:

- **proposition** - Atomic propositions extracted from documents
- **entity** - Local entities (before entity linking)
- **global_entity** - Global entities after fusion

Edge types:
- **Skeleton edges** (Nucleus-Nucleus): SEQUENCE, CONTRAST, CONCESSION
- **Detail edges** (Nucleus-Satellite): CAUSED_BY, MOTIVATION, ELABORATION, BACKGROUND
- **Mention edges**: MENTIONS_ENTITY (proposition <-> entity, bidirectional)

### Key Components

#### 1. Graph Building (`src/proposition_graph/`)

- **`graph_builder.py`**: Main class that constructs the graph from documents
- **`unified_extractor.py`**: Extracts propositions, entities, and RST relations in a single LLM call
- **`rst_analyzer.py`**: Rhetorical Structure Theory analysis for proposition relationships

#### 2. Entity Linking (`src/entity_linking/`)

Two-stage process:
- **Stage 1 (`2a-generate_candidates.py`)**: Uses embedding similarity to generate candidate entity groups
- **Stage 2 (`2b-link_and_fuse.py`)**: Uses LLM to validate candidates and fuse graph

Key classes:
- **`CandidateGenerator`**: Generates candidate entity groups using vector similarity
- **`EntityLinker`**: LLM-based entity linking validation
- **`GraphFusion`**: Merges linked entities into global entity nodes

#### 3. Online Retrieval (`src/retrieval/`)

**Agent State Machine** (`agent_state_machine.py`):
```
CHECK_PLAN -> RETRIEVE -> MAP -> UPDATE -> (CHECK_PLAN or ANSWER)
```

- **CHECK_PLAN**: LLM checks if current info is sufficient, identifies gaps
- **RETRIEVE**: Finds anchor points using vector search
- **MAP**: Beam search for path exploration (progressive)
- **UPDATE**: Ranks paths, extracts evidence
- **ANSWER**: Generates final answer

**Path Scoring** (`path_scorer.py`):
```
S(v) = w_sem * S_sem(v) + w_bridge * S_bridge(v)

S_sem = cosine_similarity(embed(question), embed(node))
S_bridge = log(1 + new_entities) / log(1 + global_norm)
```

**Key optimizations**:
- `EmbeddingCacheManager`: Memory-based LRU cache for embeddings
- Adjacency cache: Pre-built neighbor mapping for O(1) neighbor lookup
- Batch embedding: Combines multiple `embed_single` calls into batch API

#### 4. Configuration (`src/config/`)

- **`model_config.py`**: LLM and embedding model settings
- **`retrieval_config.py`**: Search parameters (beam width, max rounds, weights)
- **`graph_config.py`**: Graph construction settings

Configuration is loaded from `config.yaml` and accessible via global getter functions.

### Performance Considerations

**Known optimizations**:
- Use `--index-dir` and `--persistence-dir` flags for faster initialization
- Two-stage entity linking when GPU memory is limited
- Batch processing in QA uses shared resources (initialized once, reused)

**Timing logs**: Enable by setting `verbose=True` when running `agent_machine.run()`. See `docs/TIMING_GUIDE.md`.

**Embedding calls**: The system uses batch embedding APIs when available. The `EmbeddingCacheManager` (in-memory cache, no persistence) significantly reduces API calls.

### Data Flow

**Offline Phase (vLLM offline inference)**:
1. Documents → `UnifiedDocumentExtractor` (vLLM) → Propositions, Entities, RST Relations
2. Graph → `CandidateGenerator` (vLLM embedding) → Candidate entity groups
3. Candidate groups → `EntityLinker` (vLLM) → Fusion decisions
4. Raw graph + decisions → `GraphFusion` → Linked graph
5. Save to: `output/{dataset}/proposition_graph/linked_graph.pkl`

**Online Phase (API requests)**:
1. Load pre-built graph and optional indices
2. Question → `AgentStateMachine.CHECK_PLAN` (OpenAIClient API) → Info gaps
3. Gaps → `AgentStateMachine.RETRIEVE` (OpenAIEmbeddingClient API) → Anchor points (vector search)
4. Anchors → `AgentStateMachine.MAP` → Beam search exploration
5. Paths → `AgentStateMachine.UPDATE` → Ranked paths, evidence
6. Evidence → `AgentStateMachine.ANSWER` (OpenAIClient API) → Final answer

### Important Implementation Notes

1. **Bidirectional edges**: MENTIONS_ENTITY edges are always bidirectional. RST edges have a `direction` attribute indicating traversal direction. SIMILARITY edges are always bidirectional.

2. **Entity deduplication**: Entities are deduplicated within documents using `(doc_id, text)` as key during graph building.

3. **Graph format**: Graphs are saved in both `.pkl` (pickle) and `.json` (node_link_data) formats for flexibility.

4. **Configuration**: Model paths in `config.yaml` are environment-specific and need to be updated when running on different machines.

5. **Async/Await**: The entire codebase is async-based. LLM and embedding clients use async APIs. Batch processing uses `asyncio.gather` for concurrency.

6. **Resource management**: For batch QA, resources are initialized once and shared across concurrent tasks to minimize startup time and memory usage.
