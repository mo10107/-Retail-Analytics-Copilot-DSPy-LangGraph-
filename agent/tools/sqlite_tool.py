import sqlite3
import pandas as pd
import os

# Get the absolute path to the database
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "northwind.sqlite")

def get_schema():
    """Returns detailed schema using PRAGMA for live schema inspection."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_parts = ["Database Schema (SQLite):\n"]
        
        for table in tables:
            # Get table info using PRAGMA
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            schema_parts.append(f"\n{table}:")
            for col in columns:
                col_id, name, col_type, not_null, default_val, pk = col
                pk_marker = " (PRIMARY KEY)" if pk else ""
                schema_parts.append(f"  - {name} ({col_type}){pk_marker}")
        
        # Add common joins info
        schema_parts.append("\n\nRevenue Calculation: UnitPrice * Quantity * (1 - Discount)")
        schema_parts.append("\nCommon Joins:")
        schema_parts.append("  - orders JOIN order_items ON orders.OrderID = order_items.OrderID")
        schema_parts.append("  - order_items JOIN products ON order_items.ProductID = products.ProductID")
        schema_parts.append("  - products JOIN categories ON products.CategoryID = categories.CategoryID (to get CategoryName)")
        schema_parts.append("  - orders JOIN customers ON orders.CustomerID = customers.CustomerID")
        
        schema_parts.append("\n\nIMPORTANT SQLite Syntax:")
        schema_parts.append("  - Use strftime('%Y', OrderDate) for year extraction")
        schema_parts.append("  - Use strftime('%m', OrderDate) for month extraction")
        schema_parts.append("  - Date filtering: OrderDate >= '1997-01-01' AND OrderDate <= '1997-12-31'")
        schema_parts.append("  - NEVER use BETWEINTERVAL, DATEPART, or YEAR() - use strftime() instead")
        schema_parts.append("  - CategoryName is in categories table, NOT products - always JOIN categories!")
        
        conn.close()
        return "\n".join(schema_parts)
        
    except Exception as e:
        # Fallback to basic schema if PRAGMA fails
        return """
Database Schema (SQLite):

categories (CategoryID, CategoryName, Description)
products (ProductID, ProductName, CategoryID, UnitPrice, UnitsInStock, Discontinued)
customers (CustomerID, CompanyName, ContactName, Country, City, Region)
orders (OrderID, CustomerID, OrderDate, RequiredDate, ShippedDate, ShipCountry, Freight)
order_items (OrderID, ProductID, UnitPrice, Quantity, Discount)

Revenue Calculation: UnitPrice * Quantity * (1 - Discount)
"""

def execute_sql(query: str):
    """Executes SQL and returns results with table names used."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Extract table names from query for citation tracking
        query_upper = query.upper()
        tables_used = []
        table_keywords = ['FROM', 'JOIN']
        
        for keyword in table_keywords:
            if keyword in query_upper:
                # Simple extraction - look for table names after keywords
                parts = query_upper.split(keyword)
                for part in parts[1:]:
                    # Get first word after keyword (the table name)
                    words = part.strip().split()
                    if words:
                        table_name = words[0].strip('(),;').lower()
                        # Capitalize first letter for consistency
                        table_name = table_name.replace('_', ' ').title().replace(' ', '_')
                        if table_name and table_name not in tables_used:
                            tables_used.append(table_name)
        
        return {
            "success": True, 
            "data": df.to_dict(orient="records"), 
            "columns": list(df.columns),
            "tables_used": tables_used
        }
    except Exception as e:
        return {
            "success": False, 
            "error": str(e),
            "tables_used": []
        }