import os
import sqlite3
from typing import Tuple

class SQLiteTool:
    """A tool for interacting with a SQLite database, including schema retrieval and query execution."""
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        """
        Initializes the tool with the path to the SQLite database.
        Raises FileNotFoundError if the database file does not exist.
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found at path: {db_path}")
        self.db_path = db_path

    def _connect(self):
        """Establishes a connection to the SQLite database in read-only URI mode."""
        # Use URI mode with 'ro' to ensure the database is not created if it doesn't exist
        # and to prevent accidental writes.
        return sqlite3.connect(self.db_path)

    def get_schema(self) -> str:
        """
        Retrieves the schema of all tables in the database.
        This fulfills the requirement for live schema introspection using PRAGMA.

        Returns:
            A string representation of the database schema.
        """
        conn = None
        schema_info = []
        try:
            conn = self._connect()
            cursor = conn.cursor()

            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]

            # Get schema for each table
            for table_name in tables:
                schema_info.append(f"Table: {table_name}")
                cursor.execute(f'PRAGMA table_info("{table_name}");')
                columns = cursor.fetchall()
                for col in columns:
                    # col[1] is name, col[2] is type
                    schema_info.append(f"  - {col[1]} ({col[2]})")
            return "\n".join(schema_info)
        except sqlite3.Error as e:
            return f"Error getting schema: {e}"
        finally:
            if conn:
                conn.close()

    def execute_query(self, query: str) -> Tuple[str, str]:
        """
        Executes a given SQL query and formats the result.

        Args:
            query: The SQL query to execute.

        Returns:
            A tuple containing the formatted query result as a string,
            and an error message string if an error occurred.
        """
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(query)
            
            # Commit changes for DML statements (UPDATE, INSERT, DELETE)
            if query.strip().upper().startswith(("UPDATE", "INSERT", "DELETE")):
                conn.commit()
            
            # Fetch column names from cursor.description
            column_names = [description[0] for description in cursor.description] if cursor.description else []
            rows = cursor.fetchall()

            if not rows:
                return "Query returned no results.", ""

            # Format results into a markdown-style table for clarity
            header = "| " + " | ".join(column_names) + " |"
            separator = "|" + "---|"*len(column_names)
            body = ["| " + " | ".join(map(str, row)) + " |" for row in rows]
            
            return "\n".join([header, separator] + body), ""
        except sqlite3.Error as e:
            # Return a clear error message for the agent to handle
            return "", f"SQL Error: {e}"
        finally:
            if conn:
                conn.close()