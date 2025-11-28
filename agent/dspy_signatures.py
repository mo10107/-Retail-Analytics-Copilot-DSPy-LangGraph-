import dspy
from typing import Optional, List

# 1. Router: Decides strategy [cite: 100]
class RouteQuery(dspy.Signature):
    """Classify the user question into a strategy: 'sql', 'rag', or 'hybrid'."""
    question = dspy.InputField()
    strategy = dspy.OutputField(desc="One of: sql, rag, hybrid")

# 2. Planner: Extract constraints from RAG context
class ExtractConstraints(dspy.Signature):
    """Extract structured constraints from retrieved documents for SQL generation.
    Identify: date ranges, KPI formulas, category names, entity IDs."""
    
    question = dspy.InputField()
    rag_context = dspy.InputField(desc="Retrieved document chunks")
    
    date_ranges = dspy.OutputField(desc="Extracted date ranges (e.g., '1997-06-01 to 1997-06-30')")
    kpi_formulas = dspy.OutputField(desc="Relevant KPI calculation formulas")
    categories = dspy.OutputField(desc="Product categories or entity names mentioned")
    constraints_summary = dspy.OutputField(desc="Brief summary of all constraints")

# 3. NL to SQL: The component we will optimize [cite: 103, 112]
class GenerateSQL(dspy.Signature):
    """Generate VALID SQLite query. CRITICAL: Use SQLite functions only!
    - Date year: strftime('%Y', OrderDate) = '1997' 
    - Date month: strftime('%m', OrderDate) = '06'
    - Date range: OrderDate >= '1997-01-01' AND OrderDate <= '1997-12-31'
    - NEVER use DATEPART, YEAR(), MONTH(), BETWEINTERVAL - these are NOT SQLite functions!
    Tables: orders, order_items, products, customers, categories"""
    
    schema_context = dspy.InputField(desc="Table schemas and relations")
    question = dspy.InputField()
    constraints = dspy.InputField(desc="Extracted constraints (dates, KPIs, categories)")
    sql_query = dspy.OutputField(desc="Valid SQLite query. No markdown. Use strftime() for dates.")

# 4. Synthesizer: Produces final typed answer [cite: 105]
class SynthesizeAnswer(dspy.Signature):
    """Answer the question based on SQL results and Context. 
    Return a precise answer matching the format hint."""
    
    question = dspy.InputField()
    sql_query = dspy.InputField()
    sql_result = dspy.InputField()
    rag_context = dspy.InputField()
    format_hint = dspy.InputField(desc="e.g. 'float', 'int', 'list[dict]'")
    
    final_answer = dspy.OutputField(desc="The answer matching the format_hint type")
    explanation = dspy.OutputField(desc="Brief explanation < 2 sentences")
    citations = dspy.OutputField(desc="List of tables used (e.g. Orders) and doc chunk IDs")
    