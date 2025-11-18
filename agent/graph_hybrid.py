import operator
import re
from typing import Annotated, List, Literal, TypedDict

import dspy
from langgraph.graph import END, StateGraph

# These imports are correct for the project structure and assume you run the application
# from the root directory (e.g., python run_agent_hybrid.py)
from agent.dspy_signatures import NLSQL, Router, Synthesizer
from agent.rag.retrieval import Retriever 
from agent.tools.sqlite_tool import SQLiteTool

# --- Constants ---
OLLAMA_MODEL = "phi3.5:3.8b-mini-instruct-q2_K"
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_SQL_RETRIES = 2
DEFAULT_RETRIEVAL_K = 3
DB_TABLES = ["Orders", "Order Details", "Products", "Customers", "Categories", "Suppliers"]

# --- Agent State ---
class AgentState(TypedDict):
    """Defines the complete state for the agent's workflow."""
    question: str
    format_hint: str
    context: Annotated[List[str], operator.add]
    sql_query: str
    sql_result: str
    final_answer: str
    citations: Annotated[List[str], operator.add]
    error: str
    num_sql_retries: int
    category: str
    retrieval_scores: List[float]
    sql_success: bool

# --- DSPy and Tool Initialization ---
# Initialize DSPy with Ollama
# try:
#     lm = dspy.OllamaLocal(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
# except AttributeError:
#     # Fallback for different DSPy versions
#     lm = dspy.LM(model=f"ollama/{OLLAMA_MODEL}", api_base=OLLAMA_BASE_URL)
def use_gemini_for_fast_non_local_reponse_testing():
    import os
    import dotenv
    dotenv.load_dotenv()

    return dspy.LM(model="gemini/gemini-flash-latest", api_key=os.environ["GEMINI_API_KEY"])

lm = use_gemini_for_fast_non_local_reponse_testing()
dspy.settings.configure(lm=lm)

router_module = Router()
nl_sql_module = NLSQL()
synthesizer_module = Synthesizer()
retriever_module = Retriever()
sqlite_tool = SQLiteTool()

# --- Helper Functions ---
def extract_tables_from_sql(sql_query: str) -> List[str]:
    """Extracts database table names from an SQL query for citation."""
    # This regex is a bit more robust for finding table names, including quoted ones.
    pattern = r'\bFROM\s+([`"\'\w]+)|\bJOIN\s+([`"\'\w]+)'
    matches = re.findall(pattern, sql_query, re.IGNORECASE)
    
    found_tables = set()
    raw_tables = [item for sublist in matches for item in sublist if item]
    
    for raw_table in raw_tables:
        clean_table = raw_table.strip('`"\'')
        # Check against the canonical list of DB tables
        for db_table in DB_TABLES:
            if db_table.lower() == clean_table.lower():
                found_tables.add(db_table)
                
    return list(found_tables)

def parse_citations(raw_citations) -> List[str]:
    """Robustly parses citations from various string formats (list, csv)."""
    if isinstance(raw_citations, list):
        return [str(c).strip() for c in raw_citations if c]
    citations_str = str(raw_citations).strip().strip('"\'[]')
    # Split by comma or newline
    return [c.strip() for c in re.split(r'[,\n]', citations_str) if c.strip()]

# --- Node Functions ---

def route_question(state: AgentState) -> dict:
    """Node 1: Classifies the question into 'rag', 'sql', or 'hybrid'."""
    print("---NODE: Route Question---")
    try:
        prediction = router_module(question=state["question"])
        category = prediction.category.lower().strip()
        if category not in ["rag", "sql", "hybrid"]:
            category = "hybrid" # Default to hybrid for safety
        print(f"  - Category: {category}")
        return {"category": category, "error": ""}
    except Exception as e:
        print(f"  - Router Error: {e}, defaulting to hybrid.")
        return {"category": "hybrid", "error": f"Router failed: {e}"}

def retrieve_documents(state: AgentState) -> dict:
    """Node 2: Retrieves relevant document chunks using the retriever."""
    print("---NODE: Retrieve Documents---")
    try:
        contents, citations = retriever_module.retrieve(state["question"], k=DEFAULT_RETRIEVAL_K)
        print(f"  - Retrieved {len(citations)} chunks.")
        return {"context": contents, "citations": citations, "retrieval_scores": [1.0] * len(contents)}
    except Exception as e:
        print(f"  - Retrieval Error: {e}")
        return {"context": [], "citations": [], "error": f"Retrieval failed: {e}"}

def plan_execution(state: AgentState) -> dict:
    """Node 3: Planner to extract constraints (for logging and potential use)."""
    print("---NODE: Plan Execution---")
    context_text = " ".join(state.get("context", []))
    dates = re.findall(r'\d{4}-\d{2}-\d{2}', context_text)
    if dates:
        print(f"  - Extracted dates: {dates}")
    return {}

def generate_sql(state: AgentState) -> dict:
    """Node 4: Generates an SQL query using DSPy NL->SQL module. Uses sqlite3. don't use markdown."""
    print("---NODE: Generate SQL---")
    try:
        schema = sqlite_tool.get_schema()
        
        enhanced_question = state["question"]
        if state.get("context") and state["category"] == "hybrid":
            context_summary = " ".join(state["context"])
            enhanced_question = f"Use the following context to inform the query:\n{context_summary}\n\nQuestion: {state['question']}"
        
        prediction = nl_sql_module(question=enhanced_question, schema=schema)
        sql_query = getattr(prediction, 'sql_query', '').strip()
        
        # Clean up common LLM artifacts
        sql_query = re.sub(r'```sql\s*|\s*```', '', sql_query)
        if ';' in sql_query:
            sql_query = sql_query.split(';')[0] + ';'
        elif sql_query:
            sql_query += ';'
            
        print(f"  - Generated SQL: {sql_query}")
        
        tables_used = extract_tables_from_sql(sql_query)
        print(f"  - Tables used for citation: {tables_used}")
        
        return {"sql_query": sql_query, "citations": tables_used, "error": ""}
    except Exception as e:
        print(f"  - SQL Generation Error: {e}")
        return {"sql_query": "", "error": f"SQL generation failed: {e}", "num_sql_retries": state.get("num_sql_retries", 0) + 1}

def execute_sql(state: AgentState) -> dict:
    """Node 5: Executes the SQL query using the SQLite tool."""
    print("---NODE: Execute SQL---")
    if not state.get("sql_query"):
        return {"sql_result": "", "error": "No SQL query to execute", "sql_success": False}
    
    try:
        result, error = sqlite_tool.execute_query(state["sql_query"])
        if error:
            print(f"  - SQL Execution Error: {error}")
            return {"sql_result": "", "error": error, "sql_success": False, "num_sql_retries": state.get("num_sql_retries", 0) + 1}
        
        print(f"  - SQL executed successfully.")
        return {"sql_result": result, "error": "", "sql_success": True}
    except Exception as e:
        print(f"  - SQL Execution Exception: {e}")
        return {"sql_result": "", "error": str(e), "sql_success": False, "num_sql_retries": state.get("num_sql_retries", 0) + 1}

def synthesize_answer(state: AgentState) -> dict:
    """Node 6: Synthesizes the final answer using context and SQL results."""
    print("---NODE: Synthesize Answer---")
    try:
        prediction = synthesizer_module(
            question=state["question"],
            context=state.get("context", []),
            sql_result=state.get("sql_result", ""),
            format_hint=state.get("format_hint", "")
        )
        
        final_answer = getattr(prediction, 'answer', '').strip()
        new_citations = parse_citations(getattr(prediction, 'citations', ''))
        
        print(f"  - Synthesized Answer: {final_answer}")
        print(f"  - Parsed Citations: {new_citations}")
        
        # Combine citations from all sources (retrieval, SQL gen)
        return {"final_answer": final_answer, "citations": new_citations, "error": ""}
    except Exception as e:
        print(f"  - Synthesis Error: {e}")
        return {"final_answer": "", "error": f"Synthesis failed: {e}"}

# --- Conditional Logic and Repair Loop ---

def should_generate_sql(state: AgentState) -> str:
    """Conditional Edge: Determines if SQL generation is needed."""
    return "generate_sql" if state["category"] in ["sql", "hybrid"] else "synthesize_answer"

def decide_repair_strategy(state: AgentState) -> Literal["generate_sql", "synthesize_answer", "__end__"]:
    """Conditional Edge: Implements the required repair loop."""
    print("---NODE: Decide Repair Strategy---")
    error = state.get("error", "")
    num_retries = state.get("num_sql_retries", 0)

    # Condition 1: SQL failed and we have retries left
    if "SQL" in error and num_retries < MAX_SQL_RETRIES:
        print(f"  - Decision: Repairing SQL (Attempt {num_retries + 1})")
        return "generate_sql"
    
    # Condition 2: Synthesis failed or produced an empty answer. Retry synthesis once.
    if not state.get("final_answer") and num_retries < 1:
        print("  - Decision: Retrying synthesis due to empty answer.")
        return "synthesize_answer"
        
    print("  - Decision: Ending graph execution.")
    return "__end__"

# --- Graph Construction ---
def build_graph() -> StateGraph:
    """Constructs and compiles the complete LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("plan_execution", plan_execution)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("synthesize_answer", synthesize_answer)
    
    # Define the graph's flow
    workflow.set_entry_point("route_question")
    
    workflow.add_conditional_edges(
        "route_question",
        lambda state: state["category"],
        {"rag": "retrieve_documents", "sql": "generate_sql", "hybrid": "retrieve_documents"}
    )
    
    workflow.add_edge("retrieve_documents", "plan_execution")
    
    workflow.add_conditional_edges(
        "plan_execution",
        should_generate_sql,
        {"generate_sql": "generate_sql", "synthesize_answer": "synthesize_answer"}
    )
    
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_edge("execute_sql", "synthesize_answer")
    
    # The final node either ends or loops back for repair
    workflow.add_conditional_edges(
        "synthesize_answer",
        decide_repair_strategy,
    )
    
    return workflow.compile()

# --- Exportable App and Confidence Score ---
app = build_graph()

def calculate_confidence(state: dict) -> float:
    """Calculates a confidence score based on workflow execution signals."""
    confidence = 0.5  # Base confidence

    if state.get("context"):
        confidence += 0.1
    if state.get("sql_success", False):
        confidence += 0.2
    if state.get("sql_result") and "no results" not in state["sql_result"]:
        confidence += 0.1
    if len(state.get("citations", [])) > 1:
        confidence += 0.1

    # Penalize for errors and retries
    num_retries = state.get("num_sql_retries", 0)
    confidence -= 0.15 * num_retries
    
    if state.get("error"):
        confidence -= 0.3
    
    return max(0.0, min(1.0, confidence))