import streamlit as st
import sqlite3
import pandas as pd
import re
import os
import tempfile
from typing import Dict, List, Optional, Any
import json
from io import StringIO

# Configure page
st.set_page_config(
    page_title="Universal Text to SQL Converter",
    page_icon="🗄️",
    layout="wide"
)

class UniversalTextToSQLConverter:
    def __init__(self):
        self.sql_keywords = {
            # Selection keywords
            'show': 'SELECT', 'display': 'SELECT', 'get': 'SELECT', 'find': 'SELECT',
            'list': 'SELECT', 'retrieve': 'SELECT', 'fetch': 'SELECT', 'give': 'SELECT',
            
            # Aggregation keywords
            'count': 'COUNT', 'total': 'COUNT(*)', 'sum': 'SUM', 'add': 'SUM',
            'average': 'AVG', 'mean': 'AVG', 'maximum': 'MAX', 'minimum': 'MIN',
            'max': 'MAX', 'min': 'MIN', 'highest': 'MAX', 'lowest': 'MIN',
            
            # Condition keywords
            'where': 'WHERE', 'having': 'HAVING', 'group by': 'GROUP BY',
            'order by': 'ORDER BY', 'sort': 'ORDER BY', 'arrange': 'ORDER BY',
            
            # Comparison operators
            'greater than': '>', 'more than': '>', 'above': '>', 'over': '>',
            'less than': '<', 'fewer than': '<', 'below': '<', 'under': '<',
            'equal to': '=', 'equals': '=', 'is': '=', 'equal': '=',
            'not equal': '!=', 'not equals': '!=', 'different': '!=',
            'contains': 'LIKE', 'includes': 'LIKE', 'has': 'LIKE',
            'starts with': 'LIKE', 'begins with': 'LIKE', 'ends with': 'LIKE',
            
            # Logical operators
            'and': 'AND', 'or': 'OR', 'not': 'NOT',
            
            # Other keywords
            'all': '*', 'everything': '*', 'distinct': 'DISTINCT', 'unique': 'DISTINCT',
            'top': 'LIMIT', 'first': 'LIMIT', 'limit': 'LIMIT',
        }
    
    def detect_table_from_text(self, text: str, tables: List[str]) -> Optional[str]:
        """Detect which table is being referenced in the text"""
        text_lower = text.lower()
        
        # Direct table name match
        for table in tables:
            if table.lower() in text_lower:
                return table
        
        # Fuzzy matching for common table name patterns
        table_keywords = {
            'employee': ['employee', 'staff', 'worker', 'person', 'people'],
            'customer': ['customer', 'client', 'user', 'buyer'],
            'product': ['product', 'item', 'goods', 'merchandise'],
            'order': ['order', 'purchase', 'transaction', 'sale'],
            'payment': ['payment', 'transaction', 'billing'],
            'invoice': ['invoice', 'bill', 'receipt'],
        }
        
        for table in tables:
            table_lower = table.lower()
            for category, keywords in table_keywords.items():
                if category in table_lower:
                    for keyword in keywords:
                        if keyword in text_lower:
                            return table
        
        return None
    
    def detect_columns_from_text(self, text: str, table_columns: List[str]) -> List[str]:
        """Detect which columns are being referenced"""
        text_lower = text.lower()
        mentioned_columns = []
        
        for col in table_columns:
            col_lower = col.lower()
            if col_lower in text_lower:
                mentioned_columns.append(col)
        
        # Common column name mappings
        column_mappings = {
            'name': ['name', 'title', 'label'],
            'id': ['id', 'identifier', 'key'],
            'price': ['price', 'cost', 'amount', 'value'],
            'date': ['date', 'time', 'when'],
            'salary': ['salary', 'wage', 'pay', 'income'],
            'age': ['age', 'years'],
            'email': ['email', 'mail'],
            'phone': ['phone', 'number', 'contact'],
        }
        
        for col in table_columns:
            col_lower = col.lower()
            for category, keywords in column_mappings.items():
                if category in col_lower:
                    for keyword in keywords:
                        if keyword in text_lower and col not in mentioned_columns:
                            mentioned_columns.append(col)
        
        return mentioned_columns
    
    def extract_conditions(self, text: str, table_columns: List[str]) -> List[str]:
        """Extract WHERE conditions from natural language"""
        conditions = []
        text_lower = text.lower()
        
        # Extract numeric conditions
        numeric_patterns = [
            (r'(\w+)\s+(?:greater than|more than|above|over)\s+(\d+)', '>'),
            (r'(\w+)\s+(?:less than|fewer than|below|under)\s+(\d+)', '<'),
            (r'(\w+)\s+(?:equal to|equals|is)\s+(\d+)', '='),
            (r'(\w+)\s+(?:not equal|not equals|different)\s+(\d+)', '!='),
        ]
        
        for pattern, operator in numeric_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                column_word, value = match
                # Find matching column
                for col in table_columns:
                    if column_word in col.lower() or col.lower() in column_word:
                        conditions.append(f"{col} {operator} {value}")
                        break
        
        # Extract string conditions
        string_patterns = [
            (r'(\w+)\s+(?:is|equals|equal to)\s+["\']([^"\']+)["\']', '='),
            (r'(\w+)\s+(?:contains|includes|has)\s+["\']([^"\']+)["\']', 'LIKE'),
            (r'(\w+)\s+(?:starts with|begins with)\s+["\']([^"\']+)["\']', 'LIKE'),
        ]
        
        for pattern, operator in string_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                column_word, value = match
                for col in table_columns:
                    if column_word in col.lower() or col.lower() in column_word:
                        if operator == 'LIKE':
                            if 'contains' in text_lower or 'includes' in text_lower:
                                conditions.append(f"{col} LIKE '%{value}%'")
                            elif 'starts with' in text_lower:
                                conditions.append(f"{col} LIKE '{value}%'")
                            elif 'ends with' in text_lower:
                                conditions.append(f"{col} LIKE '%{value}'")
                        else:
                            conditions.append(f"{col} = '{value}'")
                        break
        
        return conditions
    
    def parse_natural_language(self, text: str, tables: List[str], columns: Dict[str, List[str]]) -> str:
        """Convert natural language to SQL query"""
        text = text.strip()
        text_lower = text.lower()
        
        # Debug: Add logging to see what's happening
        print(f"DEBUG: Input text: '{text}'")
        print(f"DEBUG: Available tables: {tables}")
        print(f"DEBUG: Available columns: {columns}")
        
        # Detect table - improved logic
        detected_table = None
        
        # First try exact table name match
        for table in tables:
            if table.lower() in text_lower:
                detected_table = table
                break
        
        # If no exact match, try fuzzy matching
        if not detected_table:
            table_keywords = {
                'employee': ['employee', 'staff', 'worker', 'person', 'people', 'emp'],
                'product': ['product', 'item', 'goods', 'merchandise', 'prod'],
                'order': ['order', 'purchase', 'transaction', 'sale'],
                'customer': ['customer', 'client', 'user', 'buyer'],
            }
            
            for table in tables:
                table_lower = table.lower()
                # Check if any keyword matches the table name
                for category, keywords in table_keywords.items():
                    if category in table_lower:
                        for keyword in keywords:
                            if keyword in text_lower:
                                detected_table = table
                                break
                        if detected_table:
                            break
                if detected_table:
                    break
        
        # If still no table, infer from column names
        if not detected_table:
            for table, cols in columns.items():
                for col in cols:
                    if col.lower() in text_lower:
                        detected_table = table
                        break
                if detected_table:
                    break
        
        # Default to first table if none detected
        if not detected_table and tables:
            detected_table = tables[0]
        
        if not detected_table:
            return "-- ERROR: No table found. Available tables: " + ", ".join(tables)
        
        print(f"DEBUG: Detected table: {detected_table}")
        
        table_columns = columns[detected_table]
        
        # Build SELECT clause
        select_part = "SELECT *"
        mentioned_columns = []
        
        # Look for specific columns mentioned
        for col in table_columns:
            if col.lower() in text_lower:
                mentioned_columns.append(col)
        
        # If specific columns mentioned, use them
        if mentioned_columns:
            select_part = f"SELECT {', '.join(mentioned_columns)}"
        
        print(f"DEBUG: Mentioned columns: {mentioned_columns}")
        
        # Handle aggregation functions - more comprehensive
        aggregation_found = False
        
        if any(word in text_lower for word in ['count', 'total', 'how many', 'number of']):
            if 'distinct' in text_lower and mentioned_columns:
                select_part = f"SELECT COUNT(DISTINCT {mentioned_columns[0]})"
            else:
                select_part = "SELECT COUNT(*)"
            aggregation_found = True
            
        elif any(word in text_lower for word in ['sum', 'total of', 'add up', 'sum of']):
            # Find numeric columns for SUM
            numeric_cols = [col for col in mentioned_columns if any(num_word in col.lower() for num_word in ['price', 'salary', 'amount', 'cost', 'value', 'quantity'])]
            if numeric_cols:
                select_part = f"SELECT SUM({numeric_cols[0]})"
            elif mentioned_columns:
                select_part = f"SELECT SUM({mentioned_columns[0]})"
            aggregation_found = True
            
        elif any(word in text_lower for word in ['average', 'mean', 'avg']):
            numeric_cols = [col for col in mentioned_columns if any(num_word in col.lower() for num_word in ['price', 'salary', 'amount', 'cost', 'value', 'age', 'quantity'])]
            if numeric_cols:
                select_part = f"SELECT AVG({numeric_cols[0]})"
            elif mentioned_columns:
                select_part = f"SELECT AVG({mentioned_columns[0]})"
            aggregation_found = True
            
        elif any(word in text_lower for word in ['maximum', 'max', 'highest', 'largest']):
            if mentioned_columns:
                select_part = f"SELECT MAX({mentioned_columns[0]})"
            aggregation_found = True
            
        elif any(word in text_lower for word in ['minimum', 'min', 'lowest', 'smallest']):
            if mentioned_columns:
                select_part = f"SELECT MIN({mentioned_columns[0]})"
            aggregation_found = True
        
        # Handle DISTINCT
        if not aggregation_found and ('distinct' in text_lower or 'unique' in text_lower):
            if mentioned_columns:
                select_part = f"SELECT DISTINCT {', '.join(mentioned_columns)}"
            else:
                select_part = "SELECT DISTINCT *"
        
        # Build FROM clause
        from_part = f"FROM {detected_table}"
        
        # Build WHERE clause - improved condition detection
        where_conditions = []
        
        # Look for numeric conditions with more patterns
        numeric_patterns = [
            (r'(\w+)\s+(?:greater than|more than|above|over|>)\s+(\d+)', '>'),
            (r'(\w+)\s+(?:less than|fewer than|below|under|<)\s+(\d+)', '<'),
            (r'(\w+)\s+(?:equal to|equals|is|=)\s+(\d+)', '='),
            (r'(\w+)\s+(?:not equal|not equals|different|!=)\s+(\d+)', '!='),
            (r'(?:greater than|more than|above|over|>)\s+(\d+)', '>'),  # Without column specified
            (r'(?:less than|fewer than|below|under|<)\s+(\d+)', '<'),   # Without column specified
        ]
        
        for pattern, operator in numeric_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if len(match) == 2:  # Column and value
                    column_word, value = match
                    # Find matching column
                    matching_col = None
                    for col in table_columns:
                        if column_word in col.lower() or col.lower() in column_word:
                            matching_col = col
                            break
                    if matching_col:
                        where_conditions.append(f"{matching_col} {operator} {value}")
                else:  # Just value, find numeric column
                    value = match[0] if isinstance(match, tuple) else match
                    numeric_cols = [col for col in table_columns if any(num_word in col.lower() for num_word in ['price', 'salary', 'amount', 'cost', 'value', 'age', 'quantity', 'id'])]
                    if numeric_cols:
                        where_conditions.append(f"{numeric_cols[0]} {operator} {value}")
        
        # Look for string conditions in department, category, etc.
        department_matches = re.findall(r'(?:in|from)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)', text_lower)
        for dept in department_matches:
            dept = dept.strip()
            if len(dept) > 2:  # Avoid short words
                # Find text columns
                text_cols = [col for col in table_columns if any(text_word in col.lower() for text_word in ['department', 'category', 'type', 'status', 'name'])]
                if text_cols:
                    where_conditions.append(f"{text_cols[0]} = '{dept.title()}'")
        
        where_part = ""
        if where_conditions:
            where_part = f"WHERE {' AND '.join(where_conditions)}"
        
        print(f"DEBUG: WHERE conditions: {where_conditions}")
        
        # Handle ORDER BY
        order_by_part = ""
        if any(phrase in text_lower for phrase in ['order by', 'sort by', 'sorted by', 'arrange by', 'sort']):
            if mentioned_columns:
                direction = "DESC" if any(word in text_lower for word in ['desc', 'descending', 'highest first', 'largest first', 'high to low']) else "ASC"
                order_by_part = f"ORDER BY {mentioned_columns[0]} {direction}"
        
        # Handle LIMIT
        limit_part = ""
        limit_numbers = re.findall(r'(?:top|first|limit)\s+(\d+)', text_lower)
        if limit_numbers:
            limit_part = f"LIMIT {limit_numbers[0]}"
        elif any(word in text_lower for word in ['top 5', 'first 5']):
            limit_part = "LIMIT 5"
        elif any(word in text_lower for word in ['top 10', 'first 10']):
            limit_part = "LIMIT 10"
        
        # Construct final query
        query_parts = [select_part, from_part, where_part, order_by_part, limit_part]
        sql_query = ' '.join([part for part in query_parts if part])
        
        print(f"DEBUG: Final query: {sql_query}")
        
        return sql_query

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.db_type = None
    
    def connect_sqlite_file(self, file_path: str):
        """Connect to SQLite file"""
        try:
            self.connection = sqlite3.connect(file_path)
            self.db_type = "SQLite"
            return True, "Connected successfully to SQLite database"
        except Exception as e:
            return False, f"Error connecting to SQLite: {str(e)}"
    
    def connect_sqlite_memory(self, sql_script: str):
        """Create SQLite database in memory from SQL script"""
        try:
            self.connection = sqlite3.connect(':memory:')
            self.connection.executescript(sql_script)
            self.db_type = "SQLite"
            return True, "SQLite database created successfully in memory"
        except Exception as e:
            return False, f"Error creating SQLite database: {str(e)}"
    
    def create_sample_database(self):
        """Create a sample database for demonstration"""
        sql_script = '''
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT,
            salary INTEGER,
            age INTEGER,
            hire_date TEXT,
            email TEXT
        );
        
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock INTEGER,
            supplier TEXT
        );
        
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_name TEXT,
            product_id INTEGER,
            quantity INTEGER,
            order_date TEXT,
            status TEXT,
            FOREIGN KEY (product_id) REFERENCES products (id)
        );
        
        INSERT INTO employees VALUES 
        (1, 'John Doe', 'Engineering', 75000, 30, '2020-01-15', 'john.doe@company.com'),
        (2, 'Jane Smith', 'Marketing', 65000, 28, '2019-03-22', 'jane.smith@company.com'),
        (3, 'Bob Johnson', 'Engineering', 80000, 35, '2018-07-10', 'bob.johnson@company.com'),
        (4, 'Alice Brown', 'HR', 60000, 32, '2021-02-01', 'alice.brown@company.com'),
        (5, 'Charlie Wilson', 'Sales', 70000, 29, '2020-05-18', 'charlie.wilson@company.com'),
        (6, 'Diana Prince', 'Engineering', 85000, 33, '2019-11-30', 'diana.prince@company.com'),
        (7, 'Eve Davis', 'Marketing', 62000, 26, '2021-08-15', 'eve.davis@company.com');
        
        INSERT INTO products VALUES 
        (1, 'Laptop', 'Electronics', 999.99, 50, 'TechCorp'),
        (2, 'Mouse', 'Electronics', 25.99, 200, 'TechCorp'),
        (3, 'Keyboard', 'Electronics', 79.99, 150, 'TechCorp'),
        (4, 'Desk Chair', 'Furniture', 299.99, 75, 'FurniturePlus'),
        (5, 'Monitor', 'Electronics', 349.99, 100, 'TechCorp'),
        (6, 'Desk Lamp', 'Furniture', 49.99, 120, 'FurniturePlus'),
        (7, 'Webcam', 'Electronics', 89.99, 80, 'TechCorp');
        
        INSERT INTO orders VALUES 
        (1, 'Michael Brown', 1, 2, '2023-01-15', 'Delivered'),
        (2, 'Sarah Johnson', 3, 1, '2023-01-20', 'Shipped'),
        (3, 'David Lee', 2, 5, '2023-02-01', 'Processing'),
        (4, 'Lisa Chen', 5, 1, '2023-02-10', 'Delivered'),
        (5, 'Tom Wilson', 4, 2, '2023-02-15', 'Shipped');
        '''
        
        return self.connect_sqlite_memory(sql_script)
    
    def get_database_schema(self):
        """Get database schema information"""
        if not self.connection:
            return [], {}
        
        cursor = self.connection.cursor()
        
        try:
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get columns for each table
            columns = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                table_columns = [row[1] for row in cursor.fetchall()]
                columns[table] = table_columns
            
            return tables, columns
        except Exception as e:
            st.error(f"Error getting schema: {str(e)}")
            return [], {}
    
    def execute_query(self, query: str):
        """Execute SQL query and return results"""
        if not self.connection:
            raise Exception("No database connection")
        
        return pd.read_sql_query(query, self.connection)

def main():
    st.title("🗄️ Universal Text to SQL Converter")
    st.markdown("Convert natural language queries into SQL statements for any database!")
    
    # Initialize session state
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    db_manager = st.session_state.db_manager
    
    # Database Connection Section
    st.header("📊 Database Connection")
    
    connection_tab1, connection_tab2, connection_tab3 = st.tabs(["📁 Upload Database", "💾 Sample Database", "📝 SQL Script"])
    
    with connection_tab1:
        st.subheader("Upload SQLite Database File")
        uploaded_file = st.file_uploader("Choose a SQLite database file", type=['db', 'sqlite', 'sqlite3'])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            success, message = db_manager.connect_sqlite_file(tmp_file_path)
            if success:
                st.success(message)
            else:
                st.error(message)
    
    with connection_tab2:
        st.subheader("Use Sample Database")
        st.markdown("Create a sample database with employees, products, and orders tables for testing.")
        
        if st.button("Create Sample Database", type="primary"):
            success, message = db_manager.create_sample_database()
            if success:
                st.success(message)
            else:
                st.error(message)
    
    with connection_tab3:
        st.subheader("Create Database from SQL Script")
        sql_script = st.text_area(
            "Enter SQL script to create database:",
            placeholder="CREATE TABLE example (\n    id INTEGER PRIMARY KEY,\n    name TEXT\n);\n\nINSERT INTO example VALUES (1, 'Sample');",
            height=200
        )
        
        if st.button("Create Database from Script"):
            if sql_script:
                success, message = db_manager.connect_sqlite_memory(sql_script)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Please enter a SQL script")
    
    # Get database schema
    tables, columns = db_manager.get_database_schema()
    
    if not tables:
        st.warning("⚠️ No database connected. Please connect to a database first.")
        return
    
    # Sidebar with database info
    with st.sidebar:
        st.header("📊 Database Schema")
        st.write(f"**Database Type:** {db_manager.db_type}")
        st.write(f"**Tables:** {len(tables)}")
        
        for table in tables:
            with st.expander(f"📋 {table} ({len(columns[table])} columns)"):
                for col in columns[table]:
                    st.write(f"• {col}")
        
        st.header("💡 Example Queries")
        st.markdown("""
        **Basic Queries:**
        - "Show all employees"
        - "List products"
        - "Get all orders"
        
        **Filtered Queries:**
        - "Find employees in Engineering"
        - "Show products with price greater than 100"
        - "Get orders with status 'Delivered'"
        
        **Aggregation:**
        - "Count total employees"
        - "Average salary of employees"
        - "Sum of all product prices"
        - "Maximum age of employees"
        
        **Advanced:**
        - "Show top 5 most expensive products"
        - "Get unique departments"
        - "Find employees with salary greater than 70000"
        - "List products sorted by price descending"
        """)
    
    # Main Query Interface
    st.header("🗣️ Natural Language to SQL")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Natural Language Input")
        user_query = st.text_area(
            "Enter your query in plain English:",
            placeholder="e.g., Show all employees in Engineering department with salary greater than 70000",
            height=120,
            key="nl_input"
        )
        
        # Quick example buttons
        st.write("**Quick Examples:**")
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            if st.button("👥 All Employees"):
                st.session_state.nl_input = "Show all employees"
            if st.button("💰 High Salaries"):
                st.session_state.nl_input = "Find employees with salary greater than 70000"
            if st.button("📊 Count Products"):
                st.session_state.nl_input = "Count total products"
        
        with example_col2:
            if st.button("🔍 Engineering Dept"):
                st.session_state.nl_input = "Show employees in Engineering"
            if st.button("💸 Expensive Items"):
                st.session_state.nl_input = "Show products with price greater than 100"
            if st.button("📈 Average Salary"):
                st.session_state.nl_input = "Get average salary of employees"
        
        if st.button("🔄 Convert to SQL", type="primary"):
            if user_query:
                try:
                    converter = UniversalTextToSQLConverter()
                    
                    # Show debug info
                    with st.expander("🔍 Debug Information"):
                        st.write("**Input Text:**", user_query)
                        st.write("**Available Tables:**", tables)
                        st.write("**Available Columns:**", columns)
                    
                    sql_query = converter.parse_natural_language(user_query, tables, columns)
                    st.session_state.generated_sql = sql_query
                    
                    if sql_query.startswith("-- ERROR"):
                        st.error(sql_query)
                    else:
                        st.success("✅ SQL query generated successfully!")
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating SQL: {str(e)}")
                    st.write("**Debug Info:**")
                    st.write(f"Tables: {tables}")
                    st.write(f"Columns: {columns}")
            else:
                st.warning("Please enter a query first!")
    
    with col2:
        st.subheader("Generated SQL")
        if 'generated_sql' in st.session_state:
            sql_query = st.text_area(
                "Generated SQL (editable):",
                value=st.session_state.generated_sql,
                height=120,
                key="sql_editor"
            )
        else:
            sql_query = st.text_area(
                "Generated SQL will appear here:",
                height=120,
                disabled=True,
                key="sql_placeholder"
            )
    
    # Execute Query Section
    st.header("▶️ Execute Query")
    
    if sql_query and not sql_query.startswith("--"):
        col3, col4, col5 = st.columns([1, 1, 2])
        
        with col3:
            execute_button = st.button("🚀 Execute SQL", type="secondary")
        
        with col4:
            if 'last_results' in st.session_state:
                csv = st.session_state.last_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )
        
        if execute_button:
            try:
                with st.spinner("Executing query..."):
                    df = db_manager.execute_query(sql_query)
                    st.session_state.last_results = df
                
                st.success(f"✅ Query executed successfully! Found {len(df)} rows.")
                
                # Display results
                if not df.empty:
                    st.subheader("📋 Query Results")
                    st.dataframe(df, use_container_width=True, height=400)
                    
                    # Basic statistics
                    if len(df) > 0:
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        with col_stats1:
                            st.metric("Total Rows", len(df))
                        with col_stats2:
                            st.metric("Total Columns", len(df.columns))
                        with col_stats3:
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            st.metric("Numeric Columns", len(numeric_cols))
                else:
                    st.info("Query executed successfully but returned no results.")
                    
            except Exception as e:
                st.error(f"❌ Error executing query: {str(e)}")
                st.code(sql_query, language="sql")
    
    elif sql_query and sql_query.startswith("--"):
        st.warning("⚠️ Please refine your natural language query. The system couldn't generate a valid SQL query.")
    
    # Advanced Features
    with st.expander("🔧 Advanced Features"):
        st.subheader("Query History")
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        if sql_query and sql_query not in st.session_state.query_history:
            st.session_state.query_history.append(sql_query)
        
        if st.session_state.query_history:
            selected_query = st.selectbox("Previous Queries:", st.session_state.query_history)
            if st.button("Load Selected Query"):
                st.session_state.generated_sql = selected_query
                st.rerun()
        
        st.subheader("Export Options")
        if 'last_results' in st.session_state and not st.session_state.last_results.empty:
            df = st.session_state.last_results
            
            # JSON export
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name="query_results.json",
                mime="application/json"
            )
            
            # Excel export (if openpyxl is available)
            try:
                excel_buffer = StringIO()
                df.to_excel(excel_buffer, index=False)
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_buffer.getvalue(),
                    file_name="query_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except:
                st.info("Excel export requires openpyxl package")

if __name__ == "__main__":
    main()
