import json
import click
from agent.graph_hybrid import app as agent_app, get_event_log, clear_event_log
from agent.rag.retrieval import load_and_chunk_docs

@click.command()
@click.option('--batch', required=True, help='Path to input JSONL file')
@click.option('--out', required=True, help='Path to output JSONL file')
def run(batch, out):
    """
    Main entry point for the Retail Analytics Copilot.
    """
    # 1. Initialize Retrieval Index
    print("Initializing Retrieval System...")
    load_and_chunk_docs()

    # 2. Read Input Questions
    questions = []
    with open(batch, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    print(f"Loaded {len(questions)} questions.")

    # 3. Process Questions
    results = []
    for q_data in questions:
        print(f"\nProcessing: {q_data['id']}...")
        
        # Clear event log for this question
        clear_event_log()
        
        # Prepare initial state for LangGraph
        initial_state = {
            "question": q_data['question'],
            "format_hint": q_data['format_hint'],
            "strategy": "",
            "rag_context": "",
            "rag_chunks": [],
            "constraints": "",
            "sql_query": None,
            "sql_result": None,
            "sql_error": None,
            "tables_used": [],
            "retries": 0,
            "validation_error": None,
            "confidence": 1.0,
            "final_output": {}
        }

        try:
            # Run the graph
            final_state = agent_app.invoke(initial_state)
            
            # Extract the formatted output from the state
            output_payload = final_state.get("final_output", {})
            
            # Ensure ID matches the input
            output_payload["id"] = q_data["id"]
            
            # Ensure confidence is set
            if "confidence" not in output_payload:
                output_payload["confidence"] = final_state.get("confidence", 0.5)
            
            # Ensure all required fields exist
            if "sql" not in output_payload:
                output_payload["sql"] = final_state.get("sql_query", "")
            
            if "citations" not in output_payload:
                output_payload["citations"] = []
            
            if "explanation" not in output_payload:
                output_payload["explanation"] = "No explanation available."
            
            results.append(output_payload)
            
            print(f"  [OK] Completed with confidence: {output_payload['confidence']:.2f}")
            print(f"  Citations: {', '.join(output_payload['citations'][:3])}...")
            
        except Exception as e:
            print(f"  [ERROR] {str(e)}")
            # Create error output
            error_output = {
                "id": q_data["id"],
                "final_answer": "",
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Error during processing: {str(e)}",
                "citations": []
            }
            results.append(error_output)
        
        # Save event log for debugging
        event_log = get_event_log()
        if event_log:
            log_file = f"logs/{q_data['id']}_trace.json"
            import os
            os.makedirs("logs", exist_ok=True)
            with open(log_file, 'w') as f:
                json.dump(event_log, f, indent=2)

    # 4. Write Output
    with open(out, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print(f"\n{'='*70}")
    print(f"✅ Done. Results written to {out}")
    print(f"   Processed {len(results)} questions")
    print(f"   Event logs saved to logs/ directory")
    print(f"{'='*70}")

if __name__ == "__main__":
    run()