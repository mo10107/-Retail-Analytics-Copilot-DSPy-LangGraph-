"""
DSPy Optimizer Training Script
Optimizes the NL→SQL module using BootstrapFewShot on a small training set.
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
import json
from agent.tools.sqlite_tool import execute_sql, get_schema
from agent.dspy_signatures import GenerateSQL

# Initialize DSPy with Ollama
lm = dspy.LM('ollama/phi3.5:3.8b-mini-instruct-q4_K_M', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# Training examples for NL→SQL optimization
training_examples = [
    {
        "question": "How many orders were placed in 1997?",
        "constraints": "Date range: 1997-01-01 to 1997-12-31",
        "expected_sql": "SELECT COUNT(*) FROM orders WHERE OrderDate BETWEEN '1997-01-01' AND '1997-12-31'"
    },
    {
        "question": "Total revenue from Beverages category?",
        "constraints": "Categories: Beverages",
        "expected_sql": "SELECT SUM(oi.UnitPrice * oi.Quantity * (1 - oi.Discount)) FROM order_items oi JOIN products p ON oi.ProductID = p.ProductID JOIN categories c ON p.CategoryID = c.CategoryID WHERE c.CategoryName = 'Beverages'"
    },
    {
        "question": "Top 3 customers by order count?",
        "constraints": "",
        "expected_sql": "SELECT c.CompanyName, COUNT(o.OrderID) as OrderCount FROM customers c JOIN orders o ON c.CustomerID = o.CustomerID GROUP BY c.CompanyName ORDER BY OrderCount DESC LIMIT 3"
    },
    {
        "question": "Average order value in June 1997?",
        "constraints": "Date range: 1997-06-01 to 1997-06-30\nKPI: AOV = SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID)",
        "expected_sql": "SELECT SUM(oi.UnitPrice * oi.Quantity * (1 - oi.Discount)) / COUNT(DISTINCT oi.OrderID) FROM orders o JOIN order_items oi ON o.OrderID = oi.OrderID WHERE o.OrderDate BETWEEN '1997-06-01' AND '1997-06-30'"
    },
    {
        "question": "Which category had highest quantity sold in Summer 1997?",
        "constraints": "Date range: 1997-06-01 to 1997-06-30\nCategories: all",
        "expected_sql": "SELECT c.CategoryName, SUM(oi.Quantity) as TotalQty FROM orders o JOIN order_items oi ON o.OrderID = oi.OrderID JOIN products p ON oi.ProductID = p.ProductID JOIN categories c ON p.CategoryID = c.CategoryID WHERE o.OrderDate BETWEEN '1997-06-01' AND '1997-06-30' GROUP BY c.CategoryName ORDER BY TotalQty DESC LIMIT 1"
    }
]

# Convert to DSPy Examples
dspy_examples = []
for ex in training_examples:
    example = dspy.Example(
        schema_context=get_schema(),
        question=ex["question"],
        constraints=ex["constraints"],
        sql_query=ex["expected_sql"]
    ).with_inputs("schema_context", "question", "constraints")
    dspy_examples.append(example)

# Validation metric: SQL executes successfully
def validate_sql(example, pred, trace=None):
    """Check if generated SQL is valid and executes."""
    try:
        result = execute_sql(pred.sql_query)
        # SQL must execute successfully
        if result['success']:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        return 0.0

def train_sql_generator():
    """Train the SQL generation module."""
    
    print("="*70)
    print("DSPy OPTIMIZER TRAINING - NL to SQL Module")
    print("="*70)
    
    # Baseline: unoptimized module - use Predict instead of ChainOfThought for better parsing
    print("\n1. Testing BASELINE (unoptimized)...")
    baseline_module = dspy.Predict(GenerateSQL)
    
    baseline_success = 0
    for ex in dspy_examples[:5]:  # Test on training set
        pred = baseline_module(
            schema_context=ex.schema_context,
            question=ex.question,
            constraints=ex.constraints
        )
        score = validate_sql(ex, pred)
        baseline_success += score
        status = "✓" if score > 0 else "✗"
        print(f"  {status} {ex.question[:50]}... | Score: {score}")
    
    baseline_accuracy = baseline_success / len(dspy_examples[:5])
    print(f"\n  Baseline Accuracy: {baseline_accuracy:.2%} ({int(baseline_success)}/{len(dspy_examples[:5])} successful)")
    
    # Optimize using BootstrapFewShot
    print("\n2. OPTIMIZING with BootstrapFewShot...")
    print("   (This may take a few minutes with local LLM)")
    
    optimizer = BootstrapFewShot(
        metric=validate_sql,
        max_bootstrapped_demos=3,
        max_labeled_demos=3
    )
    
    try:
        optimized_module = optimizer.compile(
            dspy.ChainOfThought(GenerateSQL),
            trainset=dspy_examples[:5]
        )
        
        # Test optimized module
        print("\n3. Testing OPTIMIZED module...")
        optimized_success = 0
        for ex in dspy_examples[:5]:
            pred = optimized_module(
                schema_context=ex.schema_context,
                question=ex.question,
                constraints=ex.constraints
            )
            score = validate_sql(ex, pred)
            optimized_success += score
            status = "✓" if score > 0 else "✗"
            print(f"  {status} {ex.question[:50]}... | Score: {score}")
        
        optimized_accuracy = optimized_success / len(dspy_examples[:5])
        print(f"\n  Optimized Accuracy: {optimized_accuracy:.2%} ({int(optimized_success)}/{len(dspy_examples[:5])} successful)")
        
        # Results summary
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)
        print(f"Baseline:  {baseline_accuracy:.2%} ({int(baseline_success)}/{len(dspy_examples[:5])} successful)")
        print(f"Optimized: {optimized_accuracy:.2%} ({int(optimized_success)}/{len(dspy_examples[:5])} successful)")
        improvement = optimized_accuracy - baseline_accuracy
        print(f"Improvement: {improvement:+.2%}")
        print("="*70)
        
        # Save optimized module
        optimized_module.save("agent/optimized_sql_generator.json")
        print("\n✅ Optimized module saved to: agent/optimized_sql_generator.json")
        
        return optimized_module
        
    except Exception as e:
        print(f"\nOptimization failed: {str(e)}")
        print("   This is expected with local LLMs that may have context limits.")
        print(f"   Baseline performance: {baseline_accuracy:.2%}")
        return baseline_module

if __name__ == "__main__":
    train_sql_generator()
