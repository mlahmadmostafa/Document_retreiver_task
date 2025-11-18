import sqlite3
import os

DB_PATH = "data/northwind.sqlite"

def inspect_database():
    """Connects to the database and inspects its content."""
    if not os.path.exists(DB_PATH):
        print(f"Database file not found at: {DB_PATH}")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 1. Check date range in Orders table
        cursor.execute("SELECT MIN(OrderDate), MAX(OrderDate) FROM Orders;")
        date_range = cursor.fetchone()
        print("--- Date Range in Orders Table ---")
        if date_range:
            print(f"Min Order Date: {date_range[0]}")
            print(f"Max Order Date: {date_range[1]}")
        else:
            print("No data found in Orders table.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    inspect_database()
