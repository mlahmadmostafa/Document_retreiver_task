import dspy
from agent.dspy_signatures import NLSQL
from agent.tools.sqlite_tool import SQLiteTool

OLLAMA_MODEL = "phi3.5:3.8b-mini-instruct-q4_K_M"
OLLAMA_BASE_URL = "http://localhost:11434"

# For this debug file, we will use a more powerful model to ensure the prompt is the issue
llm = dspy.LM(model=f"ollama/{OLLAMA_MODEL}", api_base=OLLAMA_BASE_URL)
dspy.settings.configure(lm=llm)


def main():
    """
    Debug the NLSQL module by running it with a single question and schema.
    """
    # Get the schema
    sqlite_tool = SQLiteTool()
    schema = sqlite_tool.get_schema()

    # The question to debug
    question = "Top 3 products by total revenue all-time. Revenue uses Order Details: SUM(UnitPrice*Quantity*(1-Discount)). Return list[{product:str, revenue:float}]."

    # The NL-to-SQL module
    nl_sql_module = NLSQL()

    # Run the module
    prediction = nl_sql_module(question=question, schema=schema)

    # Print the generated SQL
    print("Generated SQL:")
    print(prediction.sql_query)


if __name__ == "__main__":
    main()