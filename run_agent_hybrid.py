import json
import re
from typing import Any, Dict, List

import click
import dspy
from dspy.teleprompt import BootstrapFewShot

# Import the compiled graph, state, and other necessary components from the agent package
from agent.graph_hybrid import AgentState, app, calculate_confidence
from agent.dspy_signatures import NLSQL
from agent.tools.sqlite_tool import SQLiteTool


class NLSQLOptimizer:
    """
    Handles the optimization of the NL->SQL DSPy module.
    This class encapsulates the training data, evaluation metric, and optimization process.
    """
    
    def __init__(self):
        """Initializes the optimizer and the SQLite tool needed for validation."""
        self.sqlite = SQLiteTool()
    
    def create_training_set(self) -> List[dspy.Example]:
        """
        Creates a small, high-quality training set for NL->SQL optimization.
        These examples are designed to teach the model common SQL patterns for the Northwind database.
        """
        schema = self.sqlite.get_schema()
        
        return [
            # Example 1: Basic COUNT
            dspy.Example(
                question="How many customers are there in the database?",
                schema=schema,
                sql_query='SELECT COUNT(*) FROM Customers;'
            ).with_inputs("question", "schema"),
            
            # Example 2: JOIN with aggregation and aliasing, correct quoting for "Order Details"
            dspy.Example(
                question="What is the total revenue for each product?",
                schema=schema,
                sql_query='SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as TotalRevenue FROM Products p JOIN "Order Details" od ON p.ProductID = od.ProductID GROUP BY p.ProductName;'
            ).with_inputs("question", "schema"),
            
            # Example 3: Date range filtering using BETWEEN
            dspy.Example(
                question="List all orders placed between June 1, 1997 and June 30, 1997.",
                schema=schema,
                sql_query="SELECT OrderID, OrderDate FROM Orders WHERE OrderDate BETWEEN '1997-06-01' AND '1997-06-30';"
            ).with_inputs("question", "schema"),
            
            # Example 4: Top N query with ORDER BY and LIMIT
            dspy.Example(
                question="Who are the top 3 customers by number of orders?",
                schema=schema,
                sql_query='SELECT c.CompanyName, COUNT(o.OrderID) as OrderCount FROM Customers c JOIN Orders o ON c.CustomerID = o.CustomerID GROUP BY c.CompanyName ORDER BY OrderCount DESC LIMIT 3;'
            ).with_inputs("question", "schema"),

            # Example 5: Quoted table name "Order Details"
            dspy.Example(
                question="What is the total revenue?",
                schema=schema,
                sql_query='SELECT SUM(UnitPrice * Quantity * (1 - Discount)) FROM "Order Details";'
            ).with_inputs("question", "schema"),
            
            # New Example 6: Using julianday for date difference (for DATEDIFF replacement)
            dspy.Example(
                question="What is the maximum number of days between order date and shipped date?",
                schema=schema,
                sql_query="SELECT MAX(julianday(ShippedDate) - julianday(OrderDate)) FROM Orders WHERE ShippedDate IS NOT NULL;"
            ).with_inputs("question", "schema"),

            # New Example 7: Using strftime for year extraction (for YEAR replacement)
            dspy.Example(
                question="How many orders were placed in 1997?",
                schema=schema,
                sql_query="SELECT COUNT(OrderID) FROM Orders WHERE strftime('%Y', OrderDate) = '1997';"
            ).with_inputs("question", "schema"),

            # New Example 8: Ambiguous column name resolution with aliases
            dspy.Example(
                question="What are the order details for order ID 10248?",
                schema=schema,
                sql_query='SELECT od.ProductID, od.UnitPrice, od.Quantity FROM "Order Details" od JOIN Orders o ON od.OrderID = o.OrderID WHERE o.OrderID = 10248;'
            ).with_inputs("question", "schema"),

            # New Example 9: Correct GROUP BY with ProductName
            dspy.Example(
                question="List the total quantity sold for each product.",
                schema=schema,
                sql_query='SELECT p.ProductName, SUM(od.Quantity) FROM Products p JOIN "Order Details" od ON p.ProductID = od.ProductID GROUP BY p.ProductName;'
            ).with_inputs("question", "schema"),
        ]
    
    def metric(self, example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
        """
        Defines the success metric for SQL optimization: does the generated SQL execute?
        Returns 1.0 for executable SQL, 0.0 otherwise.
        """
        pred_sql = getattr(prediction, 'sql_query', '').strip()
        pred_sql = re.sub(r'```sql\s*|\s*```', '', pred_sql) # Clean markdown
        
        if not pred_sql:
            return 0.0
        
        _, error = self.sqlite.execute_query(pred_sql)
        
        if error:
            print(f"  ✗ SQL Error: {str(error)[:120]}")
            return 0.0
        
        print("  ✓ Valid SQL")
        return 1.0
    
    def optimize(self) -> NLSQL:
        """
        Runs the full optimization process and returns the improved DSPy module.
        """
        click.echo("\n" + "="*60)
        click.echo("🚀 OPTIMIZING NL->SQL MODULE 🚀")
        click.echo("="*60)
        
        trainset = self.create_training_set()
        unoptimized_module = NLSQL()
        
        # Evaluate performance before optimization
        click.echo("\n--- Evaluating BEFORE optimization ---")
        evaluator = dspy.evaluate.Evaluate(devset=trainset, metric=self.metric, num_threads=1, display_progress=True)
        evaluation_results_before = evaluator(unoptimized_module)
        before_score = evaluation_results_before.score
        
        # Configure and run the teleprompter
        click.echo("\n--- Running BootstrapFewShot optimization ---")
        teleprompter = BootstrapFewShot(metric=self.metric, max_bootstrapped_demos=2)
        optimized_module = teleprompter.compile(unoptimized_module, trainset=trainset)
        
        # Evaluate performance after optimization
        click.echo("\n--- Evaluating AFTER optimization ---")
        evaluation_results_after = evaluator(optimized_module)
        after_score = evaluation_results_after.score
        
        # Print summary
        click.echo("\n" + "="*60)
        click.echo("✅ OPTIMIZATION SUMMARY ✅")
        click.echo("="*60)
        click.echo(f"Metric: SQL Executability")
        click.echo(f"Before: {before_score:.2%}")
        click.echo(f"After:  {after_score:.2%}")
        click.echo(f"Change: {after_score - before_score:+.2%}")
        click.echo("="*60 + "\n")
        
        return optimized_module


def format_output(question_id: str, final_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats the final agent state to match the strict output contract.
    """
    citations = final_state.get("citations", [])
    # Clean up citations: ensure list of unique strings
    if isinstance(citations, list):
        unique_citations = sorted(list(set(map(str, citations))))
    else:
        unique_citations = [str(citations)] if citations else []
    
    # Generate a concise explanation
    category = final_state.get("category", "unknown")
    if final_state.get("error"):
        explanation = f"The agent encountered an error: {final_state['error'][:100]}."
    elif category == "hybrid":
        explanation = "The answer was synthesized from document context and a database query."
    elif category == "sql":
        explanation = "The answer was derived directly from a database query."
    else: # rag
        explanation = "The answer was retrieved directly from the provided documents."

    return {
        "id": question_id,
        "final_answer": final_state.get("final_answer", ""),
        "sql": final_state.get("sql_query", ""),
        "confidence": round(calculate_confidence(final_state), 2),
        "explanation": explanation,
        "citations": unique_citations
    }


@click.command()
@click.option("--batch", type=click.Path(exists=True), required=True, help="Path to the input JSONL file.")
@click.option("--out", type=click.Path(), required=True, help="Path to the output JSONL file.")
@click.option("--optimize", is_flag=True, default=False, help="Run DSPy optimization before processing.")
def main(batch: str, out: str, optimize: bool):
    """
    Main CLI entrypoint to run the Retail Analytics Copilot.
    """
    # Import locally to allow module patching during optimization
    from agent import graph_hybrid

    if optimize:
        optimizer = NLSQLOptimizer()
        # The optimized module replaces the one in the imported graph_hybrid module
        graph_hybrid.nl_sql_module = optimizer.optimize()
    
    try:
        with open(batch, "r", encoding="utf-8") as f_in, open(out, "w", encoding="utf-8") as f_out:
            questions = [json.loads(line) for line in f_in]
            click.echo(f"Loaded {len(questions)} questions from '{batch}'. Processing...")

            for i, q_data in enumerate(questions, 1):
                click.echo("\n" + "-"*60)
                click.echo(f"Processing [{i}/{len(questions)}]: {q_data.get('id')}")
                click.echo(f"Q: {q_data.get('question')}")
                click.echo("-"*60)
                
                initial_state = AgentState(
                    question=q_data["question"],
                    format_hint=q_data["format_hint"],
                    context=[], sql_query="", sql_result="", final_answer="",
                    citations=[], error="", num_sql_retries=0, category="",
                    retrieval_scores=[], sql_success=False
                )
                
                final_state_snapshot = None
                for state_update in app.stream(initial_state):
                    final_state_snapshot = state_update

                # The last update contains the final state from the last-run node
                final_state = list(final_state_snapshot.values())[0] if final_state_snapshot else {}
                
                result = format_output(q_data["id"], final_state)
                f_out.write(json.dumps(result) + "\n")

                click.echo(f"  -> Answer: {result['final_answer']}")
                click.echo(f"  -> Confidence: {result['confidence']:.2f}")

            click.echo("\n" + "="*60)
            click.echo(f"✅ Batch processing complete. Results saved to '{out}'.")
            click.echo("="*60)

    except Exception as e:
        click.echo(f"\n❌ An unexpected error occurred: {e}", err=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()