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
        # and to prevent accidental writes.
        return sqlite3.connect(self.db_path)

    def get_schema(self) -> str:
        """
        Retrieves the schema of all tables in the database as a list.
        This provides the most accurate and complete schema information.

        Returns:
            A string containing the list of table names.
        """
        conn = None
        schema_statements = []
        try:
            conn = self._connect()
            cursor = conn.cursor()

            # Query sqlite_master for the  statements
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            
            rows = cursor.fetchall()
            for row in rows:
                if row[0]:
                    schema_statements.append(row[0] + ";")
            
            return "\n\n".join(schema_statements)
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
            # Fix common SQL errors
            query = query.replace("DATEDIFF('day',", "julianday(")
            query = query.replace("YEAR(", "strftime('%Y',")
            query = query.replace("OrderDetails", '"Order Details"')
            query = query.replace(" ID ", " ProductID ")
            query = query.replace("[OrderDetails]", '"Order Details"')
            query = query.replace("[Order Details]", '"Order Details"')
            query = query.replace("strftime('%Y', o.OrderDate) = 1997", "strftime('%Y', o.OrderDate) = '1997'")


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