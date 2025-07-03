import os
import json
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from database import db

# Load environment variables
load_dotenv('.env')

class DatabaseChain:
    def __init__(self):
        # Initialize OpenAI client with API key from environment variable
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("MODEL")
        )
        
        # Initialize database connection
        self.db = db
        
        # Initialize the chain
        self.chain = self._build_chain()
        
        # Session-level customer name cache
        self.customer_name_cache = {}

    def _get_customer_name(self, customer_id: str) -> str:
        """Get customer name from customer_profile table with session-level caching."""
        try:
            # Check cache first
            if customer_id in self.customer_name_cache:
                return self.customer_name_cache[customer_id]
            
            # Query database for customer name - using customer_id column
            query = f"SELECT name FROM customer_profile WHERE customer_id = '{customer_id}'"
            result = self.db.run(query)
            
            if result:
                # Parse result - it could be a string representation or list
                if isinstance(result, str):
                    import ast
                    try:
                        parsed_result = ast.literal_eval(result)
                        if parsed_result and len(parsed_result) > 0:
                            name = str(parsed_result[0][0]) if isinstance(parsed_result[0], tuple) else str(parsed_result[0])
                        else:
                            name = ""
                    except:
                        name = ""
                elif isinstance(result, list) and len(result) > 0:
                    name = str(result[0][0]) if isinstance(result[0], tuple) else str(result[0])
                else:
                    name = ""
                
                # Cache the result
                self.customer_name_cache[customer_id] = name
                return name
            else:
                # Cache empty result to avoid repeated queries
                self.customer_name_cache[customer_id] = ""
                return ""
                
        except Exception as e:
            print(f"Warning: Could not fetch customer name for {customer_id}: {e}")
            # Cache empty result to avoid repeated queries
            self.customer_name_cache[customer_id] = ""
            return ""

    def _get_schema(self, db: SQLDatabase) -> str:
        """Get schema information for raw_transactions table only using direct SQL query."""
        try:
            # Query only raw_transactions table schema directly
            schema_query = "DESCRIBE raw_transactions"
            result = self.db.run(schema_query)
            
            # Format the schema information for the LLM
            schema_lines = [
                "Table: raw_transactions",
                "",
                "Columns:"
            ]
            
            if result:
                # Parse the DESCRIBE result
                if isinstance(result, str):
                    import ast
                    try:
                        parsed_result = ast.literal_eval(result)
                        for row in parsed_result:
                            if isinstance(row, tuple) and len(row) >= 2:
                                column_name = row[0]
                                data_type = row[1]
                                nullable = row[2] if len(row) > 2 else ""
                                key = row[3] if len(row) > 3 else ""
                                default = row[4] if len(row) > 4 else ""
                                
                                # Format column info
                                column_info = f"- {column_name} ({data_type})"
                                if key == "PRI":
                                    column_info += " PRIMARY KEY"
                                if nullable == "NO":
                                    column_info += " NOT NULL"
                                
                                schema_lines.append(column_info)
                    except:
                        schema_lines.append("- Schema parsing error, but table exists")
                
                elif isinstance(result, list):
                    for row in result:
                        if isinstance(row, tuple) and len(row) >= 2:
                            column_name = row[0]
                            data_type = row[1]
                            nullable = row[2] if len(row) > 2 else ""
                            key = row[3] if len(row) > 3 else ""
                            
                            # Format column info
                            column_info = f"- {column_name} ({data_type})"
                            if key == "PRI":
                                column_info += " PRIMARY KEY"
                            if nullable == "NO":
                                column_info += " NOT NULL"
                            
                            schema_lines.append(column_info)
            
            # Add explicit note about table restriction
            schema_lines.extend([
                "",
                "IMPORTANT: This is the ONLY table available for queries.",
                "You must use 'raw_transactions' as the table name.",
                "No other tables exist in this context."
            ])
            
            return "\n".join(schema_lines)
            
        except Exception as e:
            # Fallback with basic structure
            return f"""
Table: raw_transactions

Columns:
- transaction_id (Primary Key)
- customer_id (Customer identifier)
- merchant_name (Merchant name)
- transaction_date (Transaction date in dd-mm-yyyy format)
- transaction_month (Month in YYYY-MM format)
- transaction_amount (Amount in decimal)
- transaction_type (Income/Expense)
- merchant_category (Category of merchant)
- is_online (Online/Offline indicator)
- merchant_city (City of merchant)
- merchant_logo (Merchant logo URL)
- transaction_currency (Currency code)

IMPORTANT: This is the ONLY table available for queries.
You must use 'raw_transactions' as the table name.
No other tables exist in this context.

Schema query error: {str(e)}
"""

    def _get_unique_values(self) -> Dict[str, List[str]]:
        """Get unique values for key categorical fields to help the LLM use exact values."""
        try:
            unique_values = {}
            
            # Define the categorical fields we want unique values for
            fields_to_sample = [
                'merchant_category',
                'transaction_type',
                'merchant_city',
                'transaction_currency',
                'is_online'
            ]
            
            for field in fields_to_sample:
                try:
                    # Get unique values for this field (limit to prevent overwhelming the prompt)
                    if field == 'merchant_name':
                        # For merchant names, limit to top 50 by transaction count
                        query = f"""
                        SELECT {field}, COUNT(*) as transaction_count 
                        FROM raw_transactions 
                        WHERE {field} IS NOT NULL 
                        GROUP BY {field} 
                        ORDER BY transaction_count DESC 
                        LIMIT 50
                        """
                        results = self.db.run(query)
                        # Fix: Handle the result properly for merchant names too
                        if results:
                            if isinstance(results, str):
                                import ast
                                try:
                                    parsed_results = ast.literal_eval(results)
                                    unique_values[field] = [str(row[0]) if isinstance(row, tuple) else str(row) for row in parsed_results]
                                except:
                                    unique_values[field] = []
                            elif isinstance(results, list):
                                unique_values[field] = [str(row[0]) if isinstance(row, tuple) else str(row) for row in results]
                            else:
                                unique_values[field] = []
                        else:
                            unique_values[field] = []
                    else:
                        # For other fields, get all unique values
                        query = f"SELECT DISTINCT {field} FROM raw_transactions WHERE {field} IS NOT NULL ORDER BY {field}"
                        results = self.db.run(query)
                        # Fix: Handle the result properly - it's a list of tuples, not strings
                        if results:
                            if isinstance(results, str):
                                # If it's a string representation, parse it properly
                                import ast
                                try:
                                    parsed_results = ast.literal_eval(results)
                                    unique_values[field] = [str(row[0]) if isinstance(row, tuple) else str(row) for row in parsed_results]
                                except:
                                    unique_values[field] = []
                            elif isinstance(results, list):
                                # If it's already a list, extract the first element of each tuple
                                unique_values[field] = [str(row[0]) if isinstance(row, tuple) else str(row) for row in results]
                            else:
                                unique_values[field] = []
                        else:
                            unique_values[field] = []
                        
                except Exception as field_error:
                    print(f"Warning: Could not get unique values for {field}: {field_error}")
                    unique_values[field] = []
            
            return unique_values
            
        except Exception as e:
            print(f"Warning: Could not get unique values: {e}")
            return {}

    def _format_unique_values(self, unique_values: Dict[str, List[str]]) -> str:
        """Format unique values for the prompt in a readable way."""
        if not unique_values:
            return "No unique values available."
        
        # Debug: Print merchant_category values
        if 'merchant_category' in unique_values:
            print(f"🔍 DEBUG - Found merchant categories: {unique_values['merchant_category']}")
        
        formatted = []
        for field, values in unique_values.items():
            if values:
                # Limit display to prevent overwhelming the prompt
                display_values = values[:20] if len(values) > 20 else values
                values_str = ", ".join(f"'{v}'" for v in display_values)
                if len(values) > 20:
                    values_str += f" ... (and {len(values) - 20} more)"
                formatted.append(f"- {field}: {values_str}")
            else:
                formatted.append(f"- {field}: No values found")
        
        return "\n".join(formatted)

    def _run_query(self, query: str) -> Any:
        """Execute SQL query and return results."""
        try:
            return self.db.run(query)
        except Exception as e:
            raise Exception(f"Failed to execute query: {str(e)}")

    def _run_query_with_columns(self, query: str) -> Tuple[List[str], Any]:
        """Execute SQL query and return both column names and results."""
        try:
            # Get the underlying SQLAlchemy engine
            engine = self.db._engine
            
            # Import text for raw SQL execution
            from sqlalchemy import text
            
            with engine.connect() as connection:
                # Convert string to executable SQL text object
                result = connection.execute(text(query))
                
                # Get column names from the result
                column_names = list(result.keys()) if hasattr(result, 'keys') else []
                
                # Get all rows
                rows = result.fetchall()
                
                return column_names, rows
        except Exception as e:
            raise Exception(f"Failed to execute query with columns: {str(e)}")

    def _parse_sql_results_to_json(self, column_names: List[str], sql_results: Any) -> List[Dict[str, Any]]:
        """Convert SQL results with column names to JSON format."""
        try:
            if not sql_results:
                return []
            
            data_rows = []
            
            # If we have column names and results
            if column_names and sql_results:
                for row in sql_results:
                    row_dict = {}
                    for i, column_name in enumerate(column_names):
                        if i < len(row):
                            value = row[i]
                            # Convert data types
                            row_dict[column_name] = self._convert_value_direct(value)
                    data_rows.append(row_dict)
            
            return data_rows
            
        except Exception as e:
            print(f"Error parsing SQL results: {e}")
            print(f"Column names: {column_names}")
            print(f"SQL Results: {sql_results}")
            return []
    
    def _convert_value_direct(self, value: Any) -> Any:
        """Convert value to appropriate JSON-serializable type."""
        if value is None:
            return None
        
        # Handle different data types
        if isinstance(value, (int, float, bool)):
            return value
        elif isinstance(value, str):
            return value
        else:
            # Convert other types to string
            return str(value)

    def _build_chain(self):
        """Build the LangChain pipeline."""
        # Create the prompt template with specific instructions for financial analysis
        template = """
        You are a SQL expert. Write a MySQL query to answer the user's question using the raw_transactions table.

        CRITICAL RULES:
        1. ONLY use table name "raw_transactions" 
        2. ALWAYS include: WHERE customer_id = {customer_id}
        3. Use only raw_transactions - NO other tables exist
        4. when query is specific to a date, make sure convert the date to match dd-mm-yyyy format
        4a.when you need the year use YEAR(transaction_date) to extract the year
        5. when grouping by merchant name, make sure to use  merchant_logo also so that it gets passed in the data json (not required in chart)
        6. when query is specific to show raw transactions using a filter, show transaction_date, merchant_name, transaction_amount, transaction_type, merchant_category, merchant city, is_online
        7. Use window functions (LAG, LEAD, ROW_NUMBER, RANK) for time-based comparisons and rankings
        8. Use CTEs (WITH clause) for complex calculations and multi-step analysis
        9. Alias ranking functions as 'merchant_rank' or 'spending_rank' - NEVER use 'rank' alone
        10. Handle division by zero with NULLIF: ((current - previous) / NULLIF(previous, 0)) * 100
        11. For growth rate calculations, use LAG() window function over ordered dates
        12. Use proper date ordering: ORDER BY transaction_date or ORDER BY transaction_month
        13. Include merchant_logo when aggregating by merchant for UI display
        14. Use EXACT values from available values list for categories.
        15. round off values to 1 decimal place
        16. When searching/filtering by category or merchant_name, convert both the column and the search term to lowercase and use a LIKE query with wildcards.

        Table Schema:
        {schema}

        Available Values:
        {unique_values}

        Customer: {customer_name} (ID: {customer_id})
        Question: {question}

        Write SQL query using ONLY "raw_transactions" table:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Build the SQL generation chain
        sql_chain = (
            RunnablePassthrough.assign(
                schema=lambda _: self._get_schema(self.db),
                unique_values=lambda _: self._format_unique_values(self._get_unique_values())
            )
            | prompt
            | self.llm.bind(stop=["\nSQLResult:"])
            | StrOutputParser()
        )

        # Build the response formatting chain with personalized instructions
        response_template = """You are an experienced conversational personal finance assistant to customers. 
        Based on the following SQL query and its result, provide a conversational response with insights:

        Guidelines:
        1. DO NOT display tables, data, or raw numbers in your response
        2. Use the customer's name naturally and contextually throughout the conversation, not necessarily at the beginning
        3. Use the {customer_name} in the responsesto make it  feel natural for the context, don't force it
        4. Provide 2-3 conversational lines with insights about the data in a way responding to the question
        5. Give a brief summary or key insight from the analysis
        6. End with one relevant follow-up question to continue the conversation
        7. Be friendly, helpful, and personalized
        8. Use terms like "I noticed", "It looks like", "I found that" to make it conversational
        9. Focus on insights, patterns, or interesting findings rather than listing data
        10. Weave the customer's name naturally into the response where it feels right
        11. Make it sound like a natural conversation, not a formal report

        Customer Name: {customer_name}
        SQL Query: {query}
        Result: {response}
        
        Conversational Response:"""
        response_prompt = ChatPromptTemplate.from_template(response_template)

        # Build the conversational response chain
        conversational_chain = (
            response_prompt
            | self.llm
            | StrOutputParser()
        )

        # Build the full chain that returns JSON
        def process_query(inputs):
            # Generate SQL query
            sql_query = sql_chain.invoke(inputs)
            
            # Debug: Print the generated SQL query
            print("🔍 DEBUG - Generated SQL Query:")
            print(sql_query)
            print("-" * 40)
            
            # Execute SQL query and get column names
            column_names, raw_results = self._run_query_with_columns(sql_query)
            
            # Parse results to JSON with proper column names
            json_data = self._parse_sql_results_to_json(column_names, raw_results)
            
            # Generate conversational response using the old method for display
            old_format_results = self._run_query(sql_query)
            conversational_inputs = {
                "customer_name": inputs.get("customer_name", ""),
                "query": sql_query,
                "response": old_format_results
            }
            conversational_answer = conversational_chain.invoke(conversational_inputs)
            
            # Return structured JSON
            return {
                "answer": conversational_answer,
                "data": json_data
            }

        return process_query

    def query(self, question: str, customer_id: str) -> Dict[str, Any]:
        """Execute a natural language query against the database for a specific customer."""
        try:
            # Get customer name
            customer_name = self._get_customer_name(customer_id)
            
            result = self.chain({
                "question": question,
                "customer_id": customer_id,
                "customer_name": customer_name
            })
            return result
        except Exception as e:
            return {
                "answer": f"I'm sorry, I encountered an error while analyzing your financial data: {str(e)}",
                "data": []
            }

    def _generate_chart_config(self, column_names: List[str], data: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
        """Generate chart configuration using a separate LLM call."""
        try:
            if not data or not column_names:
                return {}
            
            # Create a focused prompt for chart generation
            chart_template = """You are a data visualization expert. Based on the SQL query results, generate the optimal chart configuration.

            Analyze and use the data and provide a JSON response with chart configuration:

            Guidelines:
            1. Determine the best chart type: "line" or "bar"
            2. Identify categorical vs numerical columns
            3. For bar/line charts: categorical on x-axis, numerical on y-axis
            4. For time series: dates on x-axis, values on y-axis
            5. Create meaningful axis labels
            6. Extract actual data values from ALL the provided data (not just a sample)
            7. Use ALL data points to create complete xAxis.data and series.data arrays
            8. Ensure xAxis.data and series.data have the same length and correspond to each other

            Data columns: {column_names}
            Complete dataset: {complete_data}
            User question: {question}

            Return ONLY a valid JSON object with this exact structure:
            {{
                
                "xAxis": {{
                    "type": "category",name: "X Axis Label"
                    "data": ["value1", "value2", "value3", ...]
                }},
                "yAxis": {{
                    "type": "value", name : "Y Axis Label"
                }},
                "series": [
                    {{
                        "name": "Series Name",
                        "data": [num1, num2, num3, ...],
                        "type": "line or bar",
                        "label": {{
                            "show": true,
                            "position": "top"  // position can be: top, inside, bottom, etc.
                        }}
                    }}
                ]
                "legend": {{"show": true}},
            }}

            Important:
            - Extract ALL values from the complete dataset for xAxis.data and series.data
            - Make sure the data arrays include ALL data points, not just a few
            - Use meaningful labels for xAxis.type and yAxis.type
            - Choose appropriate chart type based on the data
            - The arrays should have the same length and represent the complete dataset
            
            Chart Configuration:"""
            
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            
            chart_prompt = ChatPromptTemplate.from_template(chart_template)
            
            # Use ALL data instead of just a sample
            complete_data = data
            
            # Create chain for chart generation
            chart_chain = chart_prompt | self.llm | StrOutputParser()
            
            # Generate chart config
            chart_response = chart_chain.invoke({
                "column_names": column_names,
                "complete_data": complete_data,
                "question": question
            })
            
            # Parse the JSON response
            import json
            import re
            
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{.*\}', chart_response, re.DOTALL)
            if json_match:
                chart_config = json.loads(json_match.group())
                
                # Validate that we have all the data points
                if chart_config.get('xAxis', {}).get('data') and chart_config.get('series'):
                    x_data_length = len(chart_config['xAxis']['data'])
                    series_data_length = len(chart_config['series'][0]['data']) if chart_config['series'] else 0
                    total_data_points = len(data)
                    
                    print(f"Chart validation: {x_data_length} x-points, {series_data_length} series points, {total_data_points} total data rows")
                
                return chart_config
            else:
                return {}
                
        except Exception as e:
            print(f"Chart generation failed: {e}")
            return {}

    def generate_chart(self, question: str, customer_id: str) -> Dict[str, Any]:
        """Execute a natural language query and generate chart in one efficient call."""
        try:
            # Step 1: Get the main query result (conversational answer + data)
            result = self.query(question, customer_id)
            
            # Step 2: Generate chart using the existing data (no re-query needed)
            chart_result = self._generate_chart_config(
                list(result['data'][0].keys()) if result.get('data') else [],
                result.get('data', []),
                question
            )
            
            # Step 3: Return combined response
            return {
                "answer": result.get('answer', ''),
                "data": result.get('data', []),
                "chart": chart_result if chart_result else {},
                "chart_status": "success" if chart_result else "no_data"
            }
            
        except Exception as e:
            return {
                "answer": f"I'm sorry, I encountered an error while analyzing your financial data: {str(e)}",
                "data": [],
                "chart": {},
                "chart_status": "error"
            }

def main():
    try:
        # Initialize the database chain
        db_chain = DatabaseChain()
        
        # Example query
        question = "what is my month wise spends in groceries"
        customer_id = "10002"
        
        print("=" * 60)
        print("🚀 FINANCIAL DATA ANALYSIS WITH PERSONALIZATION")
        print("=" * 60)
        
        # Test customer name fetching
        customer_name = db_chain._get_customer_name(customer_id)
        print(f"Customer Name: {customer_name}")
        
        # Use the efficient method that gets both data and chart in one optimized call
        result = db_chain.generate_chart(question, customer_id)
        
        print(f"Question: {question}")
        print(f"Customer ID: {customer_id}")
        print(f"Answer: {result['answer']}")
        print(f"Data Records: {result['data']}")
        print(f"Chart Status: {result['chart_status']}")
        print(f"Chart Result: {result['chart']}")
        
        if result['chart']:
            print("✅ Chart generated successfully")
            print(f"Chart has {len(result['chart'].get('xAxis', {}).get('data', []))} data points")
        else:
            print("❌ No chart generated")
        
        print("\n" + "=" * 60)
        print("✅ COMPLETE RESPONSE STRUCTURE WITH PERSONALIZATION")
        print("=" * 60)
        print(f"Response Keys: {list(result.keys())}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
