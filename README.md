# Retail Analytics Copilot - Hybrid AI Agent

A hybrid AI agent combining RAG (BM25) and SQL generation to answer business questions about Northwind retail data.

## Graph Design

**7-Node LangGraph Workflow**:
- **Router** → Classifies query strategy (`sql`, `rag`, or `hybrid`) based on question complexity
- **Retrieve** → BM25 document retrieval over 4 markdown files (marketing calendar, KPIs, catalog, policies)
- **Planner** → Extracts constraints (date ranges, KPI formulas, categories) from retrieved context to guide SQL generation
- **GenerateSQL** → NL→SQL using live schema (PRAGMA table_info) and constraints; includes SQLite-specific syntax guidance
- **ExecuteSQL** → Runs query, extracts table names for citations, handles errors
- **Validator** → Checks if answer matches `format_hint` (int/float/dict/list); triggers re-synthesis if invalid
- **Synthesize** → Produces final typed answer with explanation and citations (SQL tables + RAG doc chunks)

**Conditional paths**: RAG-only (skip SQL), Hybrid (RAG→Planner→SQL), SQL-only (skip retrieval)  
**Repair loop**: SQL errors trigger up to 2 retries with error feedback

## DSPy Optimization

**Optimized Module**: `GenerateSQL` (NL→SQL generation)  
**Optimizer**: BootstrapFewShot with 5 training examples  
**Metric**: SQL execution success rate

**Results**:
```
Baseline:     60% (3/5 successful)
Optimized:    80% (4/5 successful)
Improvement:  +20 percentage points
```

## Trade-offs & Assumptions

**1. Gross Margin**: Assumed margin percentage of 40% when not documented (CostOfGoods data unavailable in Northwind)

**2. Confidence Scoring**: Heuristic-based (1.0 base, reduced by SQL errors, retries, validation failures); not trained

**3. Campaign Dates**: "Summer Beverages 1997" = June 1-30 per `marketing_calendar.md` (database has June-Aug data available)

**4. SQLite Syntax**: Added explicit guidance to prevent common LLM errors (using `DATEPART`/`YEAR()` instead of `strftime()`, missing category JOINs)

## Running the Agent

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running with phi3.5 model
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
ollama serve

# Process batch questions
python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```

**Output Contract**: Each line in `outputs_hybrid.jsonl`:
```json
{
  "id": "question_id",
  "final_answer": "...",
  "sql": "SELECT ...",
  "confidence": 0.7,
  "explanation": "Brief explanation",
  "citations": ["Orders", "Products", "doc_chunk_id"]
}
```

## Project Structure

```
your_project/
├─ agent/
│  ├─ graph_hybrid.py          # LangGraph (7 nodes + repair loop)
│  ├─ dspy_signatures.py       # DSPy Signatures (Router/GenerateSQL/Synth)
│  ├─ rag/retrieval.py         # BM25 retrieval + chunking
│  ├─ tools/sqlite_tool.py     # DB access + PRAGMA schema
│  ├─ data/northwind.sqlite    # 1997 retail data
│  └─ docs/                    # 4 markdown files (14 chunks)
│      ├─ marketing_calendar.md
│      ├─ kpi_definitions.md
│      ├─ catalog.md
│      └─ product_policy.md
├─ sample_questions_hybrid_eval.jsonl  # Test questions
├─ run_agent_hybrid.py         # CLI entrypoint
├─ train_optimizer.py          # DSPy BootstrapFewShot training
└─ requirements.txt
```
