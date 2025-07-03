import openai
import traceback, os
from pydantic import BaseModel
from typing import List, Dict, Tuple, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sql_db_agent import DatabaseChain
import json
from travel_api_agent import get_travel_recommendations

# Pydantic Models
class ChatRequest(BaseModel):
    message: str
    customer_id: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    tool_used: Optional[str] = None
    session_id: str
    data: Optional[List[Dict[str, Any]]] = None
    chart: Optional[Dict[str, Any]] = None
    destination: Optional[Dict[str, Any]] = None
    itinerary: Optional[Dict[str, Any]] = None

# New Pydantic model for chart requests
class ChartRequest(BaseModel):
    message: str
    customer_id: str

class ChartResponse(BaseModel):
    chart: Dict[str, Any]
    status: str
    error: Optional[str] = None

# Response Parsers
class ResponseParser:
    @staticmethod
    def parse_sql_response(raw_response: Dict[str, Any], customer_id: str) -> tuple[str, Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        """Parse and format SQL agent response - returns (answer, data, chart)"""
        try:
            # The SQL agent now returns JSON with 'answer', 'data', and optionally 'chart' keys
            if isinstance(raw_response, dict):
                answer = raw_response.get('answer', 'No response available')
                data = raw_response.get('data', [])
                chart = raw_response.get('chart', {})
                
                return answer, data, chart
            else:
                # Fallback for old format
                return str(raw_response), None, None
        except Exception as e:
            return f"I encountered an issue analyzing your financial data: {str(e)}", None, None
    
    @staticmethod
    def parse_travel_response(raw_response: Dict[str, Any]) -> str:
        """Parse and format travel API response"""
        try:
            if not raw_response:
                return "I couldn't retrieve travel recommendations at the moment. Please try again later."
            
            # Extract and format the travel recommendation
            if isinstance(raw_response, dict):
                if 'recommendations' in raw_response:
                    recommendations = raw_response['recommendations']
                elif 'data' in raw_response:
                    recommendations = raw_response['data']
                else:
                    recommendations = str(raw_response)
            else:
                return f"Here are some travel recommendations for you:\n\n{str(raw_response)}"
        except Exception as e:
            return f"I had trouble processing travel recommendations: {str(e)}"

# Main Conversational Agent
class PersonalFinanceAssistant:
    def __init__(self,api_key,model):
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.7
        )
        self.model = model
        
        # Initialize OpenAI client for function calling
        self.openai_client = openai.OpenAI(api_key=api_key)
        
        # Initialize agents
        try:
            self.sql_agent = DatabaseChain()
        except Exception as e:
            print(f"Warning: SQL agent initialization failed: {e}")
            self.sql_agent = None
        
        # Session memory storage
        self.session_memories: Dict[str, ConversationBufferMemory] = {}
        
        # Session-level customer name cache
        self.session_customer_names: Dict[str, str] = {}
        
        # Session-level customer insights cache
        self.session_customer_insights: Dict[str, str] = {}
        
        # Conversation message counter for insight triggers
        self.session_message_counts: Dict[str, int] = {}
        
        # Function definitions for OpenAI
        self.functions = [
            {
                "name": "query_financial_data",
                "description": "Query user's financial transaction data, spending analysis, budgets, and financial insights. Use this for questions about spending patterns, merchant analysis, transaction history, financial summaries, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The financial question or query to analyze"
                        },
                        "customer_id": {
                            "type": "string", 
                            "description": "The customer ID to filter data for"
                        }
                    },
                    "required": ["query", "customer_id"]
                }
            },
            {
                "name": "get_travel_suggestions",
                "description": "Get travel destination recommendations and vacation planning suggestions. Use this when users want to explore travel destinations, plan trips, or get vacation ideas.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "travel_query": {
                            "type": "string",
                            "description": "The travel-related question or request"
                        }
                    },
                    "required": ["travel_query"]
                }
            }
        ]
    
    def get_session_memory(self, session_id: str) -> ConversationBufferMemory:
        """Get or create conversation memory for a session"""
        if session_id not in self.session_memories:
            self.session_memories[session_id] = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
        return self.session_memories[session_id]
    
    def get_customer_name(self, customer_id: str, session_id: str) -> str:
        """Get customer name with session-level caching"""
        # Check session cache first
        cache_key = f"{session_id}_{customer_id}"
        if cache_key in self.session_customer_names:
            return self.session_customer_names[cache_key]
        
        # Get name from SQL agent if available
        if self.sql_agent:
            try:
                customer_name = self.sql_agent._get_customer_name(customer_id)
                # Cache in session
                self.session_customer_names[cache_key] = customer_name
                return customer_name
            except Exception as e:
                print(f"Warning: Could not fetch customer name: {e}")
        
        # Cache empty result
        self.session_customer_names[cache_key] = ""
        return ""
    
    def get_customer_insights(self, customer_id: str, session_id: str) -> str:
        """Get customer insights from database with session-level caching"""
        cache_key = f"{session_id}_{customer_id}"
        
        # Check session cache first
        if cache_key in self.session_customer_insights:
            return self.session_customer_insights[cache_key]
        
        # Load from database via SQL agent
        if self.sql_agent:
            try:
                # Query customer_profile for memory_chat field
                query = f"SELECT memory_chat FROM customer_profile WHERE customer_id = '{customer_id}'"
                result = self.sql_agent.db.run(query)
                
                insights = ""
                if result:
                    if isinstance(result, str):
                        import ast
                        try:
                            parsed_result = ast.literal_eval(result)
                            if parsed_result and len(parsed_result) > 0:
                                insights = str(parsed_result[0][0]) if isinstance(parsed_result[0], tuple) else str(parsed_result[0])
                        except:
                            insights = ""
                    elif isinstance(result, list) and len(result) > 0:
                        insights = str(result[0][0]) if isinstance(result[0], tuple) else str(result[0])
                
                # Cache the result
                self.session_customer_insights[cache_key] = insights or ""
                return insights or ""
                
            except Exception as e:
                print(f"Warning: Could not fetch customer insights for {customer_id}: {e}")
                # Cache empty result
                self.session_customer_insights[cache_key] = ""
                return ""
        
        return ""
    
    def update_customer_insights(self, customer_id: str, session_id: str, new_insights: str) -> bool:
        """Update customer insights in database by overwriting with summarized version"""
        if not new_insights.strip():
            return True
        
        try:
            # Update database with the new summarized insights (overwrite, don't append)
            if self.sql_agent:
                # Use parameterized query to handle special characters
                from sqlalchemy import text
                engine = self.sql_agent.db._engine
                
                with engine.connect() as connection:
                    query = text("UPDATE customer_profile SET memory_chat = :insights WHERE customer_id = :customer_id")
                    connection.execute(query, {"insights": new_insights.strip(), "customer_id": customer_id})
                    connection.commit()
                
                # Update cache
                cache_key = f"{session_id}_{customer_id}"
                self.session_customer_insights[cache_key] = new_insights.strip()
                
                print(f"✅ Updated customer insights for {customer_id}: {new_insights}")
                return True
            
        except Exception as e:
            print(f"❌ Error updating customer insights for {customer_id}: {e}")
            return False
        
        return False
    
    def extract_customer_insights(self, customer_id: str, conversation_history: List[BaseMessage]) -> str:
        """Extract customer insights from conversation using LLM"""
        try:
            # Get only user messages from recent conversation
            user_messages = []
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                if isinstance(msg, HumanMessage):
                    user_messages.append(msg.content)
            
            if len(user_messages) < 2:
                return ""
            
            # Get existing insights for context
            existing_insights = self.get_customer_insights(customer_id, "default")
            
            # Create insight extraction prompt
            prompt = f"""
            Analyze the customer's conversation to extract high-level insights about their preferences and personality.
            
            Customer ID: {customer_id}
            Recent messages from customer: {user_messages}
            Existing insights: {existing_insights}
            
            Extract insights about:
            1. Communication preferences (brief vs detailed, visual preferences, etc.)
            2. Financial behavior patterns (budget-conscious, detail-oriented, planning-focused, etc.) 
            3. Personal interests and lifestyle (travel preferences, hobbies, etc.)
            4. Dislikes and preferences to avoid
            5. Personality traits (organized, proactive, goal-oriented, etc.)
            
            Rules:
            - Focus on USER content only, ignore assistant responses
            - Merge with existing insights to avoid repetition
            - Format as SHORT PHRASES separated by commas
            - Capture high-level behavioral patterns, not specific requests
            - Include both preferences and dislikes
            - Keep each phrase brief (2-4 words max)
            - Remove duplicate or very similar insights
            
            Format: "brief communication, budget-conscious, travel-focused, dislikes details, organized personality"
            
            If no meaningful new insights, return the existing insights in comma format.
            If no insights at all, return empty string.
            
            Customer Insights:"""
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting concise customer personality insights. Always format as brief comma-separated phrases."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            insight = response.choices[0].message.content.strip()
            
            # Clean up the response - remove any explanatory text
            if insight and len(insight) > 5:
                # Remove common prefixes and suffixes
                insight = insight.replace("Customer Insights:", "").replace("Insights:", "").strip()
                if insight.startswith('"') and insight.endswith('"'):
                    insight = insight[1:-1]
                return insight
            
            return ""
            
        except Exception as e:
            print(f"Warning: Could not extract insights: {e}")
            return ""
    
    def query_financial_data(self, query: str, customer_id: str, generate_chart: bool = False) -> tuple[str, Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        """Query financial data using SQL agent"""
        try:
            if not self.sql_agent:
                return "Financial data service is currently unavailable. Please try again later.", None, None
            
            # Get main response (answer + data)
            result = self.sql_agent.query(query, customer_id)
            answer, data, chart = ResponseParser.parse_sql_response(result, customer_id)
            
            # Optionally generate chart separately
            if generate_chart and data:
                try:
                    chart_result = self.sql_agent.generate_chart(query, customer_id)
                    chart = chart_result.get('chart', {})
                except Exception as e:
                    print(f"Chart generation failed: {e}")
                    chart = {}
            
            return answer, data, chart
            
        except Exception as e:
            return f"I'm sorry, I couldn't analyze your financial data right now. Error: {str(e)}", None, None
    
    def get_travel_suggestions(self, travel_query: str, customer_id: str, session_id: str = "default") -> tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Get travel suggestions using travel API and generate conversational response"""
        try:
            # Get structured data from travel API (now includes both destination and itinerary)
            travel_data = get_travel_recommendations(travel_query, customer_id)
            
            if not travel_data or not travel_data.get('destination'):
                return "I'm sorry, I couldn't get travel recommendations at the moment. Please try again later.", None, None
            
            # Get customer name for personalization
            customer_name = self.get_customer_name(customer_id, session_id)
            
            # Generate conversational response using OpenAI based on destination data
            conversational_response = self.generate_travel_conversation(travel_query, travel_data['destination'], customer_name)
            
            return conversational_response, travel_data.get('destination'), travel_data.get('itinerary')
            
        except Exception as e:
            error_msg = f"I'm sorry, I couldn't get travel recommendations at the moment. Error: {str(e)}"
            return error_msg, None, None
    
    def generate_travel_conversation(self, original_query: str, travel_data: Dict[str, Any], customer_name: str = "") -> str:
        """Generate natural conversational response from travel data using OpenAI"""
        try:
            # Create a prompt for generating conversational response
            prompt = f"""
            A user asked: "{original_query}"
            Customer name: {customer_name}
            
            I found this destination information from our travel API:
            {json.dumps(travel_data, indent=2)}
            
            Create a natural, engaging conversational response about this destination that:
            1. Sounds like a helpful travel assistant
            2. Uses the customer's name naturally in the conversation (not necessarily at the start)
            3. Highlights the most interesting and relevant details
            4. Includes practical information (duration, best time to visit, price level)
            5. Weaves the customer's name contextually where it feels right
            6. Ends with a helpful follow-up question
            7. Keep it to 2-3 sentences plus the question
            8. Be enthusiastic but not overly excited
            9. Make it sound like a natural conversation, not forced personalization
            
            Don't mention that this came from an API or JSON data.
            Don't force "Hey {customer_name}" unless it feels completely natural.
            """
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful travel assistant who creates natural, conversational responses about destinations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating travel conversation: {e}")
            # Fallback to basic template if OpenAI fails
            if travel_data and isinstance(travel_data, dict):
                city = travel_data.get('city', 'this destination')
                country = travel_data.get('country', '')
                description = travel_data.get('description', '')
                
                fallback_response = f"I found a great destination for you! {city}"
                if country:
                    fallback_response += f", {country}"
                if description:
                    fallback_response += f" would be perfect. {description[:100]}..."
                if customer_name:
                    fallback_response += f" Would you like me to help you plan your budget for this trip, {customer_name}?"
                else:
                    fallback_response += " Would you like me to help you plan your budget for this trip?"
                
                return fallback_response
            else:
                if customer_name:
                    return f"I found some travel recommendations for you! Would you like me to help you plan your itinerary for the trip, {customer_name}?"
                else:
                    return "I found some travel recommendations for you! Would you like me to help you plan your itinerary for the trip?"
    
    def process_function_call(self, function_call: Dict[str, Any], customer_id: str, session_id: str = "default") -> tuple[str, str, Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        """Process function calls and return response with tool used, data, chart, destination, and itinerary"""
        function_name = function_call.get("name")
        arguments = json.loads(function_call.get("arguments", "{}"))
        
        if function_name == "query_financial_data":
            answer, data, chart = self.query_financial_data(
                arguments.get("query"), 
                arguments.get("customer_id", customer_id),
                generate_chart=True  # Always generate chart for financial queries
            )
            return answer, "sql_database", data, chart, None, None
        
        elif function_name == "get_travel_suggestions":
            response, destination_data, itinerary_data = self.get_travel_suggestions(
                arguments.get("travel_query"), 
                customer_id,
                session_id
            )
            # Keep data field as None for travel queries since we have dedicated destination/itinerary fields
            return response, "travel_api", None, None, destination_data, itinerary_data
        
        else:
            return "I'm not sure how to help with that request.", "none", None, None, None, None
    
    def chat(self, message: str, customer_id: str, session_id: str) -> tuple[str, Optional[str], Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        """Main chat function with function calling and memory"""
        try:
            # Get session memory
            memory = self.get_session_memory(session_id)
            
            # Get customer name for personalization
            customer_name = self.get_customer_name(customer_id, session_id)
            name_context = f" (customer name: {customer_name})" if customer_name else ""
            
            # Get customer insights for personalization
            customer_insights = self.get_customer_insights(customer_id, session_id)
            insights_context = f"\n\nCustomer Insights: {customer_insights}" if customer_insights else ""
            
            # Track message count for insight extraction
            session_key = f"{session_id}_{customer_id}"
            if session_key not in self.session_message_counts:
                self.session_message_counts[session_key] = 0
            self.session_message_counts[session_key] += 1
            
            # Get conversation history
            chat_history = memory.chat_memory.messages if memory.chat_memory.messages else []
            
            # Prepare messages with context
            system_message = {
                "role": "system",
                "content": f"""You are a helpful personal finance assistant for customer {customer_id}{name_context}. 
                
                You can help with:
                1. Financial analysis and spending insights (use query_financial_data function)
                2. Travel recommendations and vacation planning (use get_travel_suggestions function)
                3. General financial advice and questions (answer directly)
                
                When personalizing responses:
                - Use the customer's name naturally and contextually
                - Don't force "Hey {{customer_name}}" in every response
                - Weave their name into the conversation where it feels natural
                - Make it sound like a friendly, helpful conversation
                
                Always be conversational, helpful, and personalized. When using tools, make sure to provide clear and useful analysis.{insights_context}
                """
            }
            
            # Build message history
            messages = [system_message]
            
            # Add conversation history
            for msg in chat_history[-10:]:  # Keep last 10 messages for context
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Make OpenAI call with functions
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=self.functions,
                function_call="auto",
                temperature=0.7
            )
            
            response_message = response.choices[0].message
            tool_used = None
            data = None
            chart = None
            destination = None
            itinerary = None
            
            # Check if function was called
            if response_message.function_call:
                # Process function call
                final_response, tool_used, data, chart, destination, itinerary = self.process_function_call(
                    response_message.function_call.__dict__, customer_id, session_id
                )
            else:
                # Direct response - personalization is handled by the system prompt
                final_response = response_message.content
                tool_used = None
            
            # Save to memory
            memory.chat_memory.add_user_message(message)
            # Ensure we never save None to memory - provide fallback if needed
            if final_response is None:
                final_response = "I apologize, but I couldn't generate a proper response. Please try asking again."
            memory.chat_memory.add_ai_message(final_response)
            
            # Extract insights after 2+ messages
            if self.session_message_counts[session_key] >= 2:
                try:
                    # Get updated conversation history
                    updated_history = memory.chat_memory.messages
                    new_insights = self.extract_customer_insights(customer_id, updated_history)
                    
                    if new_insights:
                        print(f"🧠 Extracted new insight for {customer_id}: {new_insights}")
                        success = self.update_customer_insights(customer_id, session_id, new_insights)
                        if not success:
                            print(f"⚠️ Failed to save insights for {customer_id}")
                
                except Exception as e:
                    print(f"❌ Error during insight extraction for {customer_id}: {e}")
            
            return final_response, tool_used, data, chart, destination, itinerary
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            print(f"Chat error: {traceback.format_exc()}")
            return error_msg, None, None, None, None, None
