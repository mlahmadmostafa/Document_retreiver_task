import pytest
import sqlite3
import os
from agent.tools.sqlite_tool import SQLiteTool

# Define a temporary database path for testing
TEST_DB_PATH = "test_northwind.db"

@pytest.fixture(scope="module")
def setup_test_db():
    """
    Fixture to set up a temporary SQLite database for testing.
    Creates a simple 'Products' table with some data.
    """
    conn = None
    try:
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Products (
                ProductID INTEGER PRIMARY KEY,
                ProductName TEXT,
                UnitPrice REAL
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Customers (
                CustomerID TEXT PRIMARY KEY,
                CompanyName TEXT
            );
        """)
        cursor.execute("INSERT INTO Products (ProductID, ProductName, UnitPrice) VALUES (1, 'Chai', 18.00);")
        cursor.execute("INSERT INTO Products (ProductID, ProductName, UnitPrice) VALUES (2, 'Chang', 19.00);")
        cursor.execute("INSERT INTO Products (ProductID, ProductName, UnitPrice) VALUES (3, 'Aniseed Syrup', 10.00);")
        cursor.execute("INSERT INTO Customers (CustomerID, CompanyName) VALUES ('ALFKI', 'Alfreds Futterkiste');")
        conn.commit()
        yield
    finally:
        if conn:
            conn.close()
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

def test_sqlite_tool_init_success(setup_test_db):
    """Test successful initialization of SQLiteTool with an existing database."""
    tool = SQLiteTool(db_path=TEST_DB_PATH)
    assert tool.db_path == TEST_DB_PATH

def test_sqlite_tool_init_file_not_found():
    """Test initialization of SQLiteTool with a non-existent database."""
    with pytest.raises(FileNotFoundError, match="Database file not found"):
        SQLiteTool(db_path="non_existent_db.db")

def test_get_schema(setup_test_db):
    """Test retrieving the schema of the test database."""
    tool = SQLiteTool(db_path=TEST_DB_PATH)
    schema = tool.get_schema()
    assert "Table: Products" in schema
    assert "  - ProductID (INTEGER)" in schema
    assert "  - ProductName (TEXT)" in schema
    assert "  - UnitPrice (REAL)" in schema
    assert "Table: Customers" in schema
    assert "  - CustomerID (TEXT)" in schema
    assert "  - CompanyName (TEXT)" in schema

def test_execute_query_select_data(setup_test_db):
    """Test executing a SELECT query that returns data."""
    tool = SQLiteTool(db_path=TEST_DB_PATH)
    query = "SELECT ProductName, UnitPrice FROM Products WHERE ProductID = 1;"
    result, error = tool.execute_query(query)
    assert error == ""
    assert "| ProductName | UnitPrice |" in result
    assert "|---|---|" in result
    assert "| Chai | 18.0 |" in result

def test_execute_query_select_no_results(setup_test_db):
    """Test executing a SELECT query that returns no results."""
    tool = SQLiteTool(db_path=TEST_DB_PATH)
    query = "SELECT ProductName FROM Products WHERE ProductID = 999;"
    result, error = tool.execute_query(query)
    assert error == ""
    assert result == "Query returned no results."

def test_execute_query_invalid_sql(setup_test_db):
    """Test executing a syntactically incorrect SQL query."""
    tool = SQLiteTool(db_path=TEST_DB_PATH)
    query = "SELECT FROM Products;" # Missing column name
    result, error = tool.execute_query(query)
    assert result == ""
    assert "SQL Error:" in error
    assert "syntax error" in error or "near \"FROM\"" in error

def test_execute_query_update_statement(setup_test_db):
    """Test executing an UPDATE statement (should return no results but no error)."""
    tool = SQLiteTool(db_path=TEST_DB_PATH)
    query = "UPDATE Products SET UnitPrice = 20.00 WHERE ProductID = 1;"
    result, error = tool.execute_query(query)
    assert error == ""
    assert result == "Query returned no results."

    # Verify the update
    query_verify = "SELECT UnitPrice FROM Products WHERE ProductID = 1;"
    result_verify, error_verify = tool.execute_query(query_verify)
    assert error_verify == ""
    assert "| UnitPrice |" in result_verify
    assert "|---|" in result_verify
    assert "| 20.0 |" in result_verify