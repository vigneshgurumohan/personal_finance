import os
from dotenv import load_dotenv
import openai
import json

# Load environment variables
load_dotenv('.env')

# Get API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
print(f"API Key loaded: {api_key[:10]}...{api_key[-10:] if api_key else 'None'}")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

# Load sample customer data
with open('summaries/customer_transaction_summaries.json') as f:
    customer_summaries = json.load(f)

# Test customer ID
customer_id = "1234"  # Using a known customer ID from the data

# Sample context
context = [
    f"Customer Summary: {json.dumps(customer_summaries[customer_id])}",
    "All currency is in AED."
]

# System prompt
system_prompt = (
    "You are a helpful finance assistant. Use the provided summaries and raw data to answer user questions clearly and concisely. "
    "If you ever present a list of merchants, categories, or any summary, you MUST use a markdown table. "
    "All currency is in AED. If a question cannot be answered from the data, say so."
)

# Test message
test_message = "What is my total income and expenses?"

try:
    # Create the full context
    full_context = system_prompt + "\n" + "\n".join(context)
    
    # Make the API call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": full_context},
            {"role": "user", "content": test_message}
        ],
        max_tokens=400
    )
    
    # Print the response
    print("\nTest Message:", test_message)
    print("\nResponse:", response.choices[0].message.content)
    
except Exception as e:
    print(f"Error: {str(e)}") 