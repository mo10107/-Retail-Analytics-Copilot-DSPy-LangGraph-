import dspy
import json
import re
from typing import TypedDict, List, Any, Optional, Dict
from langgraph.graph import StateGraph, END
from agent.dspy_signatures import RouteQuery, ExtractConstraints, GenerateSQL, SynthesizeAnswer
from agent.tools.sqlite_tool import execute_sql, get_schema
from agent.rag.retrieval import retrieve_docs

# Initialize DSPy with Ollama
lm = dspy.LM('ollama/phi3.5:3.8b-mini-instruct-q4_K_M', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# Define Modules
router = dspy.Predict(RouteQuery)
planner = dspy.Predict(ExtractConstraints)
sql_gen = dspy.Predict(GenerateSQL)
synthesizer = dspy.Predict(SynthesizeAnswer)

# Event log for tracing
EVENT_LOG = []

def log_event(node_name: str, state_snapshot: Dict):
    """Log node execution for replay/debugging."""
    event = {
        "node": node_name,
        "question": state_snapshot.get("question", ""),
        "strategy": state_snapshot.get("strategy", ""),
        "sql_query": state_snapshot.get("sql_query", ""),
        "sql_error": state_snapshot.get("sql_error", ""),
        "retries": state_snapshot.get("retries", 0)
    }
    EVENT_LOG.append(event)
    print(f"[TRACE] {node_name}: {event}")

# --- Graph State ---
class AgentState(TypedDict):
    question: str
    format_hint: str
    strategy: str  # sql, rag, hybrid
    rag_context: str
    rag_chunks: List[Dict]  # Include scores
    constraints: str  # Extracted constraints from planner
    sql_query: Optional[str]
    sql_result: Optional[Any]
    sql_error: Optional[str]
    tables_used: List[str]  # For citations
    retries: int
    validation_error: Optional[str]
    confidence: float
    final_output: dict

# --- Nodes ---

def node_router(state: AgentState):
    """Node 1: Route query to sql/rag/hybrid strategy."""
    log_event("router", state)
    pred = router(question=state['question'])
    return {"strategy": pred.strategy.lower().strip()}

def node_retrieve(state: AgentState):
    """Node 2: Retrieve top-k document chunks with scores."""
    log_event("retrieve", state)
    chunks = retrieve_docs(state['question'], top_k=3)
    
    # Format chunks into context string
    context_text = ""
    for c in chunks:
        context_text += f"[{c['id']}] (score: {c.get('score', 0):.2f}) {c['content']}\n\n"
    
    return {
        "rag_context": context_text,
        "rag_chunks": chunks
    }

def node_planner(state: AgentState):
    """Node 3: Extract structured constraints from RAG context."""
    log_event("planner", state)
    
    if not state.get('rag_context'):
        return {"constraints": "No specific constraints extracted."}
    
    pred = planner(
        question=state['question'],
        rag_context=state['rag_context']
    )
    
    # Combine extracted constraints into a summary
    constraints_text = f"""
Date Ranges: {pred.date_ranges}
KPI Formulas: {pred.kpi_formulas}
Categories/Entities: {pred.categories}
Summary: {pred.constraints_summary}
"""
    
    return {"constraints": constraints_text.strip()}

def node_generate_sql(state: AgentState):
    """Node 4: Generate SQL query using schema and constraints."""
    log_event("generate_sql", state)
    
    # Pass previous error if retrying
    error_context = ""
    if state.get('sql_error') and state.get('retries', 0) > 0:
        error_context = f"\nPrevious error: {state['sql_error']}\nPrevious query: {state.get('sql_query', '')}"
    
    constraints_input = state.get('constraints', '') + error_context
    
    pred = sql_gen(
        schema_context=get_schema(),
        question=state['question'],
        constraints=constraints_input
    )
    
    # Clean SQL (remove markdown formatting if present)
    sql_query = pred.sql_query.strip()
    sql_query = re.sub(r'```sql\s*|\s*```', '', sql_query)
    
    return {"sql_query": sql_query}

def node_execute_sql(state: AgentState):
    """Node 5: Execute SQL and capture results/errors."""
    log_event("execute_sql", state)
    
    result = execute_sql(state['sql_query'])
    
    if result['success']:
        return {
            "sql_result": result['data'],
            "sql_error": None,
            "tables_used": result.get('tables_used', [])
        }
    else:
        return {
            "sql_error": result['error'],
            "retries": state.get('retries', 0) + 1,
            "tables_used": []
        }

def node_validator(state: AgentState):
    """Node 6: Validate output format matches format_hint."""
    log_event("validator", state)
    
    final_output = state.get('final_output', {})
    final_answer = final_output.get('final_answer', '')
    format_hint = state.get('format_hint', '')
    
    validation_error = None
    
    # Validate based on format_hint
    try:
        if format_hint == 'int':
            int(final_answer)
        elif format_hint == 'float':
            float(final_answer)
        elif format_hint.startswith('{') or format_hint.startswith('list['):
            # Try to parse as JSON
            if isinstance(final_answer, str):
                json.loads(final_answer)
    except (ValueError, json.JSONDecodeError) as e:
        validation_error = f"Format validation failed: {str(e)}"
    
    return {"validation_error": validation_error}

def node_synthesize(state: AgentState):
    """Node 7: Synthesize final answer with citations."""
    log_event("synthesize", state)
    
    pred = synthesizer(
        question=state['question'],
        sql_query=state.get('sql_query', ''),
        sql_result=str(state.get('sql_result', [])),
        rag_context=state.get('rag_context', ''),
        format_hint=state['format_hint']
    )
    
    # Build comprehensive citations
    citations = []
    
    # Add DB tables used
    tables_used = state.get('tables_used', [])
    citations.extend(tables_used)
    
    # Add doc chunk IDs
    rag_chunks = state.get('rag_chunks', [])
    for chunk in rag_chunks:
        chunk_id = chunk['id']
        if chunk_id not in citations:
            citations.append(chunk_id)
    
    # Parse citations from synthesizer output if available
    if pred.citations:
        try:
            # Handle both string and list formats
            if isinstance(pred.citations, str):
                pred_cites = json.loads(pred.citations) if pred.citations.startswith('[') else [pred.citations]
            else:
                pred_cites = pred.citations
            
            for cite in pred_cites:
                if cite and cite not in citations:
                    citations.append(cite)
        except:
            pass
    
    # Calculate confidence score
    confidence = 1.0
    
    # Reduce confidence if SQL failed
    if state.get('sql_error'):
        confidence -= 0.3
    
    # Reduce confidence if retries were needed
    if state.get('retries', 0) > 0:
        confidence -= 0.2 * state.get('retries', 0)
    
    # Reduce confidence if no citations
    if not citations:
        confidence -= 0.2
    
    # Reduce confidence if validation failed
    if state.get('validation_error'):
        confidence -= 0.3
    
    confidence = max(0.0, min(1.0, confidence))
    
    return {
        "final_output": {
            "id": "unknown",  # Filled by runner
            "final_answer": pred.final_answer,
            "sql": state.get('sql_query', ''),
            "confidence": confidence,
            "explanation": pred.explanation,
            "citations": citations
        },
        "confidence": confidence
    }

# --- Edge Logic ---

def route_after_router(state: AgentState):
    """Route based on strategy."""
    strategy = state.get('strategy', 'hybrid')
    if strategy == 'rag':
        return "retrieve_rag_only"
    else:
        return "retrieve"

def route_after_retrieve(state: AgentState):
    """For RAG-only, skip to synthesize. Otherwise go to planner."""
    if state.get('strategy') == 'rag':
        return "synthesize"
    else:
        return "planner"

def check_execution(state: AgentState):
    """Check if SQL execution succeeded or needs retry."""
    if state.get('sql_error'):
        if state.get('retries', 0) < 2:
            return "retry_sql"
        else:
            return "synthesize"  # Give up, try to synthesize with partial data
    return "validator"

def check_validation(state: AgentState):
    """Check if output validation passed."""
    if state.get('validation_error') and state.get('retries', 0) < 2:
        # Could retry synthesis, but for now just proceed
        return "synthesize"
    return "end"

# --- Build Graph ---
workflow = StateGraph(AgentState)

# Add all nodes (need >= 6)
workflow.add_node("router", node_router)  # 1
workflow.add_node("retrieve", node_retrieve)  # 2
workflow.add_node("planner", node_planner)  # 3
workflow.add_node("generate_sql", node_generate_sql)  # 4
workflow.add_node("execute_sql", node_execute_sql)  # 5
workflow.add_node("validator", node_validator)  # 6
workflow.add_node("synthesize", node_synthesize)  # 7

workflow.set_entry_point("router")

# Conditional edges from router
workflow.add_conditional_edges(
    "router",
    route_after_router,
    {
        "retrieve": "retrieve",
        "retrieve_rag_only": "retrieve"
    }
)

# Conditional edges from retrieve
workflow.add_conditional_edges(
    "retrieve",
    route_after_retrieve,
    {
        "planner": "planner",
        "synthesize": "synthesize"
    }
)

# Linear flow for hybrid/sql paths
workflow.add_edge("planner", "generate_sql")
workflow.add_edge("generate_sql", "execute_sql")

# Conditional retry loop from execute_sql
workflow.add_conditional_edges(
    "execute_sql",
    check_execution,
    {
        "retry_sql": "generate_sql",
        "validator": "validator",
        "synthesize": "synthesize"
    }
)

# Validation check
workflow.add_conditional_edges(
    "validator",
    check_validation,
    {
        "synthesize": "synthesize",
        "end": "synthesize"
    }
)

workflow.add_edge("synthesize", END)

app = workflow.compile()

def get_event_log():
    """Return the event log for debugging."""
    return EVENT_LOG

def clear_event_log():
    """Clear the event log."""
    global EVENT_LOG
    EVENT_LOG = []