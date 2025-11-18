import dspy
from typing import List

# Router and NLSQL signatures remain unchanged.

# ... (Router and NLSQL code is here, no changes) ...
class RouteQuestion(dspy.Signature):
    """Classify the user's question into one of the following categories: 'rag', 'sql', or 'hybrid'."""
    question: str = dspy.InputField(desc="The user's question.")
    category: str = dspy.OutputField(desc="The category: 'rag', 'sql', or 'hybrid' (lowercase, no quotes).")

class Router(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(RouteQuestion)
    def forward(self, question: str) -> dspy.Prediction:
        return self.predictor(question=question)

class NaturalLanguageToSQL(dspy.Signature):
    """
    Your task is to generate a single, valid SQLite query to answer the user's question based on the provided schema.

    **CRITICAL INSTRUCTIONS:**
    1.  Output ONLY the SQLite query.
    2.  Do NOT include any other text, explanations, or markdown formatting like ```sql.
    3.  Ensure table names with spaces are correctly quoted (e.g., "Order Details").
    4.  The query must end with a semicolon.
    """
    question: str = dspy.InputField(desc="The user's question, potentially enriched with context from documents.")
    schema: str = dspy.InputField(desc="The complete SQLite database schema.")
    sql_query: str = dspy.OutputField(desc="A single, valid, executable SQLite query.")

class NLSQL(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(NaturalLanguageToSQL)
    def forward(self, question: str, schema: str) -> dspy.Prediction:
        return self.predictor(question=question, schema=schema)

# 3. Synthesizer Signature (MODIFICATIONS HERE)
class SynthesizeAnswer(dspy.Signature):
    """
    Synthesize a final answer to the user's question using the provided context and SQL results.

    **CRITICAL INSTRUCTIONS:**
    1.  Your final answer MUST EXACTLY match the structure and type specified in the `format_hint`.
    2.  The `sql_result` is provided as a markdown table. You must parse this table to extract the data for your answer.
    3.  When the SQL result is a single value, it will be in the second data row of the markdown table. You must extract that value directly.
    4.  Your citations MUST be a simple, comma-separated string of sources (e.g., Orders, "Order Details", marketing_calendar::chunk0).

    ---
    **EXAMPLE 1 (JSON Object Output)**

    question: "Which product category had the highest total quantity sold?"
    context: ["Marketing campaign for 'Summer Beverages 1997' ran from 1997-06-01 to 1997-06-30."]
    sql_result: '''
    | CategoryName | TotalQuantity |
    |--------------|---------------|
    | Beverages    | 5536          |
    '''
    format_hint: "{category:str, quantity:int}"

    answer: '{"category": "Beverages", "quantity": 5536}'
    citations: "Categories, Products, Order Details, marketing_calendar::chunk0"
    ---
    **EXAMPLE 2 (Single Float Output)**

    question: "What was the Average Order Value during 'Winter Classics 1997'?"
    context: ["AOV = SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID)", "Winter Classics 1997: 1997-12-01 to 1997-12-31"]
    sql_result: '''
    | ROUND(...)   |
    |--------------|
    | 1542.45      |
    '''
    format_hint: "float"

    answer: "1542.45"
    citations: "Orders, Order Details, kpi_definitions::chunk0, marketing_calendar::chunk1"
    ---
    """
    question: str = dspy.InputField(desc="The original user question.")
    context: List[str] = dspy.InputField(desc="Retrieved document chunks providing context.")
    sql_result: str = dspy.InputField(desc="Formatted results from SQL query execution, typically a markdown table.")
    format_hint: str = dspy.InputField(desc="The required output format for the answer (e.g., int, float, dict, list).")
    answer: str = dspy.OutputField(desc="The final answer, matching the format_hint exactly.")
    citations: str = dspy.OutputField(desc="A comma-separated list of data sources used.")


class Synthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SynthesizeAnswer)

    def forward(self, question: str, context: List[str], sql_result: str, format_hint: str) -> dspy.Prediction:
        return self.predictor(
            question=question, 
            context=context, 
            sql_result=sql_result, 
            format_hint=format_hint
        )