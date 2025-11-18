import pytest
from unittest.mock import patch, MagicMock
from agent.graph_hybrid import (
    router_node,
    retriever_node,
    planner_node,
    nl_to_sql_node,
    executor_node,
    synthesizer_node,
    AgentState,
)
from langchain_core.messages import HumanMessage

@patch("agent.graph_hybrid.RoutePredictor")
def test_router_node(mock_route_predictor):
    mock_predictor_instance = MagicMock()
    mock_predictor_instance.forward.return_value.route = "rag"
    mock_route_predictor.return_value = mock_predictor_instance

    state = AgentState(messages=[HumanMessage(content="test question")])
    result = router_node(state)

    assert result["route"] == "rag"
    assert result["question"] == "test question"

@patch("agent.graph_hybrid.get_retriever")
def test_retriever_node_rag_route(mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = ["doc1", "doc2"]
    mock_get_retriever.return_value = mock_retriever

    state = AgentState(route="rag", question="test question", messages=[])
    result = retriever_node(state)

    assert result["retrieved_docs"] == ["doc1", "doc2"]

def test_retriever_node_no_rag_route():
    state = AgentState(route="sql", question="test question", messages=[])
    result = retriever_node(state)
    assert result["retrieved_docs"] == []

def test_planner_node():
    state = AgentState(question="test question", messages=[])
    result = planner_node(state)
    assert result["planning_elements"] == {
        "dates": [],
        "categories": [],
        "kpis": [],
        "constraints": [],
    }

@patch("agent.graph_hybrid.get_sqlite_schema")
@patch("agent.graph_hybrid.NLtoSQLGenerator")
def test_nl_to_sql_node_sql_route(mock_nl_to_sql_generator, mock_get_sqlite_schema):
    mock_generator_instance = MagicMock()
    mock_generator_instance.forward.return_value.sql_query = "SELECT * FROM test"
    mock_nl_to_sql_generator.return_value = mock_generator_instance
    mock_get_sqlite_schema.return_value = "schema"

    state = AgentState(route="sql", question="test question", messages=[])
    result = nl_to_sql_node(state)

    assert result["sql_query"] == "SELECT * FROM test"

def test_nl_to_sql_node_no_sql_route():
    state = AgentState(route="rag", question="test question", messages=[])
    result = nl_to_sql_node(state)
    assert result["sql_query"] == ""

@patch("agent.graph_hybrid.execute_sqlite_query")
def test_executor_node_with_query(mock_execute_sqlite_query):
    mock_execute_sqlite_query.return_value = "sql result"
    state = AgentState(sql_query="SELECT * FROM test", messages=[])
    result = executor_node(state)
    assert result["sql_result"] == "sql result"

def test_executor_node_no_query():
    state = AgentState(sql_query="", messages=[])
    result = executor_node(state)
    assert result["sql_result"] == ""

@patch("agent.graph_hybrid.AnswerSynthesizer")
def test_synthesizer_node(mock_answer_synthesizer):
    mock_synthesizer_instance = MagicMock()
    mock_synthesizer_instance.forward.return_value.final_answer = "final answer"
    mock_synthesizer_instance.forward.return_value.citations = ["citation1"]
    mock_answer_synthesizer.return_value = mock_synthesizer_instance

    state = AgentState(
        question="test question",
        retrieved_docs=[MagicMock(page_content="doc content")],
        sql_result="sql result",
        messages=[]
    )
    result = synthesizer_node(state)

    assert result["final_answer"] == "final answer"
    assert result["citations"] == ["citation1"]
