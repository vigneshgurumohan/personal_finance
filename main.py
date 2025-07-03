import os
import json
import re
import traceback
from fastapi import FastAPI, Request, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import openai
from dotenv import load_dotenv
import random
from typing import List, Dict, Optional
import tiktoken
import datetime

# Load environment variables from .env file
load_dotenv('.env')

# Validate OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')

print(f"API Key loaded: {api_key}")  # Print first and last 10 chars for verification

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

client = openai.OpenAI(api_key=api_key)

# Test OpenAI connection
try:
    test_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    print("OpenAI API connection successful")
except Exception as e:
    print(f"OpenAI API connection failed: {str(e)}")
    print(traceback.format_exc())
    raise

# Load summaries at startup
try:
    with open('summaries/raw_transactions_by_customer.json') as f:
        raw_transactions = json.load(f)
    with open('summaries/customer_transaction_summaries.json') as f:
        customer_summaries = json.load(f)
    with open('summaries/top_merchants_by_category.json') as f:
        merchant_summaries = json.load(f)
    with open('summaries/top_merchants_by_month.json') as f:
        merchants_by_month = json.load(f)
    with open('summaries/insights_summary.json') as f:
        raw_insights = json.load(f)
        # Filter insights to only include essential fields
        insights_summary = {}
        for customer_id, data in raw_insights.items():
            insights_summary[customer_id] = {
                'insights': [
                    {
                        'type': insight.get('type', ''),
                        'category': insight.get('category', ''),
                        'narrative': insight.get('narrative', ''),
                        'actionable': insight.get('actionable', False),
                        'period': insight.get('period', '')
                    }
                    for insight in data.get('insights', [])
                ]
            }
        print(f"Loaded insights for {len(insights_summary)} customers")
        # Print sample insights for verification
        for customer_id, data in list(insights_summary.items())[:1]:
            print(f"\nSample insights for customer {customer_id}:")
            for insight in data['insights'][:2]:
                print(f"- Type: {insight['type']}")
                print(f"  Category: {insight['category']}")
                print(f"  Narrative: {insight['narrative'][:100]}...")
except Exception as e:
    print(f"Error loading summary files: {str(e)}")
    print(traceback.format_exc())
    raise

# In-memory conversation history per customer_id (for model-based summary)
conversation_history = {}

# Helper to extract topic from user message
merchant_keywords = set()
category_keywords = set()
# Build sets from data
for cust in merchant_summaries.values():
    for cat, merchants in cust.items():
        category_keywords.add(cat.lower())
        for m in merchants:
            merchant_keywords.add(m['merchant_name'].lower())

def extract_topic(msg):
    msg_l = msg.lower()
    found_merchants = [m for m in merchant_keywords if m in msg_l]
    found_cats = [c for c in category_keywords if c in msg_l]
    if found_merchants:
        return f"merchant: {found_merchants[0]}"
    if found_cats:
        return f"category: {found_cats[0]}"
    # fallback: first noun-like word
    match = re.search(r'\b([a-zA-Z]+)\b', msg_l)
    return match.group(1) if match else "general"

# New function to flatten JSON data into markdown tables
def flatten_to_markdown(data, data_type):
    if data_type == "customer_summary":
        # Flatten customer transaction summary
        md = "### Customer Transaction Summary\n"
        md += "| Category | Total Amount | Transaction Count | Monthly Average | % of Income | Monthly Breakdown |\n"
        md += "|----------|--------------|-------------------|-----------------|-------------|------------------|\n"
        for category, details in data.items():
            monthly_breakdown = ", ".join([f"{m}: {a} AED" for m, a in details.get('monthly_breakup', {}).items()])
            md += f"| {category} | {details.get('total', 0)} AED | {details.get('transaction_count', 0)} | {details.get('monthly_avg', 0)} AED | {details.get('pct_of_income', 0)}% | {monthly_breakdown} |\n"
        return md
    
    elif data_type == "merchant_category":
        # Flatten merchant summary by category
        md = "### Top Merchants by Category\n"
        md += "| Category | Merchant | Total Spend | % of Total | % of Category | Monthly Spend | Avg Monthly | Rank |\n"
        md += "|----------|----------|-------------|------------|---------------|---------------|-------------|------|\n"
        for category, merchants in data.items():
            for merchant in merchants[:10]:  # Only top 3 merchants per category
                monthly_spend = ", ".join([f"{m}: {s} AED" for m, s in merchant.get('monthly_spend', {}).items()])
                md += f"| {category} | {merchant['merchant_name']} | {merchant.get('total_spend', 0)} AED | {merchant.get('pct_of_total_spends', 0)}% | {merchant.get('pct_of_category_spends', 0)}% | {monthly_spend} | {merchant.get('avg_monthly_spend', 0)} AED | {merchant.get('rank', 0)} |\n"
        return md
    
    elif data_type == "merchant_month":
        # Flatten merchant summary by month
        md = "### Top Merchants by Month\n"
        md += "| Month | Merchant | Category | Total Spend | % of Monthly | Rank |\n"
        md += "|-------|----------|----------|-------------|--------------|------|\n"
        for month, merchants in data.items():
            for merchant in merchants[:10]:  # Only top 3 merchants per month
                md += f"| {month} | {merchant['merchant_name']} | {merchant.get('category', 'N/A')} | {merchant.get('total_spend', 0)} AED | {merchant.get('pct_of_total_monthly_spend', 0)}% | {merchant.get('rank', 0)} |\n"
        return md
    
    elif data_type == "conversation_summary":
        # Flatten conversation summary
        md = "### Conversation Summary\n"
        md += "| User | Bot |\n"
        md += "|------|-----|\n"
        for msg in data:
            md += f"| {msg['user']} | {msg['bot']} |\n"
        return md
    
    elif data_type == "raw_transactions":
        # Flatten raw transactions
        md = "### Raw Transactions\n"
        md += "| Date | Type | Category | Merchant | Amount | Currency | Mode |\n"
        md += "|------|------|----------|----------|--------|----------|------|\n"
        for txn in data:
            md += f"| {txn['transaction_date']} | {txn['transaction_type']} | {txn['transaction_category']} | {txn['merchant_name']} | {txn['transaction_amount']} | {txn['transaction_currency']} | {txn['transaction_mode']} |\n"
        return md
    
    return str(data)  # Fallback to string representation

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))

def detect_intent(message):
    msg = message.lower()
    # Detect month (YYYY-MM or month name, short or long)
    month_match = re.search(
        r'(\d{4}-\d{2})|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?',
        msg)
    # Detect category
    cat_match = re.search(r'groceries|entertainment|travel|utility|food|shopping|medical|salary|offline food|online food|offline shopping|offline|online', msg)
    return {
        'month': month_match.group(0) if month_match else None,
        'category': cat_match.group(0) if cat_match else None
    }

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def extract_keywords(message: str) -> Dict[str, List[str]]:
    """Extract relevant keywords from user message."""
    keywords = {
        'categories': [],
        'time_periods': [],
        'patterns': [],
        'comparisons': []
    }
    
    # Category keywords
    categories = ['groceries', 'entertainment', 'travel', 'utility', 'food', 'shopping', 
                 'medical', 'salary', 'offline food', 'online food', 'offline shopping', 
                 'offline', 'online', 'rent', 'family']
    
    # Pattern keywords
    patterns = {
        'high_spending': ['high', 'expensive', 'costly', 'spending too much', 'over budget'],
        'savings': ['saved', 'saving', 'less', 'reduced', 'decreased'],
        'comparison': ['compared', 'versus', 'than', 'more than', 'less than'],
        'anomaly': ['unusual', 'strange', 'different', 'unexpected', 'surprising']
    }
    
    # Time period keywords
    time_periods = ['month', 'week', 'year', 'last', 'previous', 'current', 'this']
    
    message_lower = message.lower()
    
    # Extract categories
    for cat in categories:
        if cat in message_lower:
            keywords['categories'].append(cat)
    
    # Extract patterns
    for pattern_type, pattern_words in patterns.items():
        for word in pattern_words:
            if word in message_lower:
                keywords['patterns'].append(pattern_type)
    
    # Extract time periods
    for period in time_periods:
        if period in message_lower:
            keywords['time_periods'].append(period)
    
    # Extract comparisons
    if any(word in message_lower for word in ['more', 'less', 'higher', 'lower', 'increase', 'decrease']):
        keywords['comparisons'].append('comparison')
    
    return keywords

def find_relevant_insights(customer_id: str, keywords: Dict[str, List[str]]) -> List[Dict]:
    """Find relevant insights based on extracted keywords."""
    print(f"\n=== Finding Relevant Insights ===")
    print(f"Customer ID: {customer_id}")
    print(f"Keywords: {keywords}")
    
    if customer_id not in insights_summary:
        print(f"No insights found for customer {customer_id}")
        return []
    
    customer_insights = insights_summary[customer_id]['insights']
    print(f"Total insights for customer: {len(customer_insights)}")
    
    # If no specific keywords, return all insights
    if not any(keywords.values()):
        print("No specific keywords, returning all insights")
        return customer_insights
    
    relevant_insights = []
    
    for insight in customer_insights:
        score = 0
        
        # Category matching
        if keywords['categories'] and insight.get('category', '').lower() in [cat.lower() for cat in keywords['categories']]:
            score += 3
            print(f"Category match found for insight: {insight.get('category')}")
        
        # Pattern matching
        if keywords['patterns']:
            if 'high_spending' in keywords['patterns'] and insight['type'] in ['anomaly', 'budget_overrun']:
                score += 2
                print(f"High spending pattern match for insight type: {insight['type']}")
            if 'savings' in keywords['patterns'] and insight['type'] == 'savings':
                score += 2
                print(f"Savings pattern match for insight type: {insight['type']}")
            if 'anomaly' in keywords['patterns'] and insight['type'] == 'anomaly':
                score += 2
                print(f"Anomaly pattern match for insight type: {insight['type']}")
        
        # Time period matching
        if keywords['time_periods'] and insight.get('period'):
            score += 1
            print(f"Time period match for insight period: {insight.get('period')}")
        
        # Comparison matching
        if keywords['comparisons'] and insight['type'] in ['trend', 'savings']:
            score += 2
            print(f"Comparison match for insight type: {insight['type']}")
        
        if score > 0:
            print(f"Adding insight with score {score}: {insight.get('type')} - {insight.get('category')}")
            relevant_insights.append((insight, score))
    
    # Sort by score and return top insights
    relevant_insights.sort(key=lambda x: x[1], reverse=True)
    print(f"Found {len(relevant_insights)} relevant insights")
    return [insight for insight, _ in relevant_insights]

def select_insight(insights: List[Dict]) -> Optional[Dict]:
    """Select an insight to show, with preference for actionable ones."""
    print(f"\n=== Selecting Insight ===")
    print(f"Total insights to select from: {len(insights)}")
    
    if not insights:
        print("No insights available to select from")
        return None
    
    # Filter actionable insights
    actionable_insights = [insight for insight in insights if insight.get('actionable', False)]
    print(f"Actionable insights: {len(actionable_insights)}")
    
    # If we have actionable insights, randomly select from those
    if actionable_insights:
        selected = random.choice(actionable_insights)
        print(f"Selected actionable insight: {selected.get('type')} - {selected.get('category')}")
        return selected
    
    # Otherwise, randomly select from all insights
    selected = random.choice(insights)
    print(f"Selected non-actionable insight: {selected.get('type')} - {selected.get('category')}")
    return selected

@app.post("/chat")
async def chat(
    customer_id: str = Form(...),
    message: str = Form(...),
    use_flattened: bool = Form(False)
):
    try:
        print(f"\n=== New Chat Request ===")
        print(f"Customer ID: {customer_id}")
        print(f"Message: {message}")
        print(f"Using flattened format: {use_flattened}")
        
        # Validate customer ID
        if customer_id not in customer_summaries:
            print(f"Invalid customer ID: {customer_id}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid customer ID: {customer_id}. Please use a valid customer ID."}
            )
        
        # Extract keywords and find relevant insights
        try:
            keywords = extract_keywords(message)
            relevant_insights = find_relevant_insights(customer_id, keywords)
        except Exception as e:
            print(f"Error in insight processing: {str(e)}")
            print(traceback.format_exc())
            keywords = {}
            relevant_insights = []
        
        # Dynamic context selection
        context = []
        try:
            intent = detect_intent(message)
            print(f"Detected intent: {intent}")
        except Exception as e:
            print(f"Error in intent detection: {str(e)}")
            print(traceback.format_exc())
            intent = {'month': None, 'category': None}
        
        # Always include memory summary if exists
        memory_summary = conversation_history.get(customer_id, {}).get('summary', "")
        if memory_summary:
            if use_flattened:
                context.append(f"Conversation summary so far:\n{memory_summary}")
            else:
                context.append(f"Conversation summary so far: {memory_summary}")
        
        # Only include customer summary if needed
        if any(word in message.lower() for word in ['summary', 'overview', 'total', 'all', 'everything']):
            try:
                if use_flattened:
                    context.append(flatten_to_markdown(customer_summaries[customer_id], "customer_summary"))
                else:
                    context.append(f"Customer Summary: {json.dumps(customer_summaries[customer_id])}")
            except Exception as e:
                print(f"Error in customer summary processing: {str(e)}")
                print(traceback.format_exc())
        
        # Add relevant merchant/category/month summaries only if specifically asked
        if intent['category'] and customer_id in merchant_summaries:
            try:
                cat = intent['category']
                cat_summary = merchant_summaries[customer_id].get(cat.title()) or merchant_summaries[customer_id].get(cat.capitalize()) or merchant_summaries[customer_id].get(cat.upper()) or merchant_summaries[customer_id].get(cat.lower())
                if cat_summary:
                    if use_flattened:
                        context.append(flatten_to_markdown({cat: cat_summary}, "merchant_category"))
                    else:
                        context.append(f"Top Merchant by category ({cat}): {json.dumps(cat_summary)}")
            except Exception as e:
                print(f"Error in category summary processing: {str(e)}")
                print(traceback.format_exc())
        
        if intent['month'] and customer_id in merchants_by_month:
            try:
                month = intent['month']
                if re.match(r'\d{4}-\d{2}', month):
                    ym = month
                else:
                    months = [
                        ('january', '01'), ('february', '02'), ('march', '03'), ('april', '04'), ('may', '05'), ('june', '06'),
                        ('july', '07'), ('august', '08'), ('september', '09'), ('october', '10'), ('november', '11'), ('december', '12'),
                        ('jan', '01'), ('feb', '02'), ('mar', '03'), ('apr', '04'), ('jun', '06'), ('jul', '07'), ('aug', '08'),
                        ('sep', '09'), ('oct', '10'), ('nov', '11'), ('dec', '12')
                    ]
                    ym = None
                    for m, mm in months:
                        if m in month:
                            ym = f"2025-{mm}"
                            break
                if ym and ym in merchants_by_month[customer_id]:
                    if use_flattened:
                        context.append(flatten_to_markdown({ym: merchants_by_month[customer_id][ym]}, "merchant_month"))
                    else:
                        context.append(f"Top Merchants by Month ({ym}): {json.dumps(merchants_by_month[customer_id][ym])}")
            except Exception as e:
                print(f"Error in month summary processing: {str(e)}")
                print(traceback.format_exc())
        
        # System prompt
        system_prompt = ("""
            "You are a very concise, very friendly, and very very very highly accurate finance assistant."
            "\n\n"
            "Data Available to You:\n"
            "- Customer summary: High-level overview of the user's spending, income, savings, and category breakdowns.\n"
            "- Merchant level summaries: Aggregated data for top merchants by category and month.\n"
            "- Relevant insights: Precomputed, user-specific insights (e.g., anomalies, trends, savings tips) provided in the context.\n"
            "- Raw transactions: The full list of individual transactions is NOT directly available to you, but the user can access them via a button.\n"
            "\n"
            "How to Answer:\n"
            "- Never hallucinate or provide random or made-up data. Only use the summaries and insights provided in the context.\n"
            "- Always dig deeper and use all relevant customer/merchant summaries and not from the insights in the context to provide a detailed breakdown or analysis.\n"
            "- For summary, breakdown, trend, or advice questions: Use the provided summaries and relevant insights to answer directly. Reference or weave insights into your answer naturally, using your own words if possible. Always Present structured numerical data as a markdown table.\n"
            "- Your answer should always include: (1) a summary or breakdown along with the period for which the data is relevant, (2) at least one relevant insight, and (3) a follow-up question or suggestion (e.g., 'Would you like to set a budget for this category?' or 'Would you like to see a merchant breakdown?').\n"
            "- Avoid one-liner answers. Be concise but thorough. Use markdown tables for clarity.\n"
            "- If you provide an insight: Integrate it smoothly into your answer. Do not simply repeat the insight verbatim unless it directly answers the user's question.\n"
            "- Always use AED as the currency.\n"
            "- If you reference a specific transaction or statistic: Do so conversationally, e.g., 'You spent 240 AED at Wooly, which is higher than your average for groceries.'\n"
            "\n"
            "Special Instructions:\n"
            "- For requests for a detailed list of transactions (e.g., 'show all my transactions in March', 'list every payment to Amazon'): DO NOT attempt to answer from summaries Instead, say : You can access your transactions by clicking the Show Raw Transactions button below.'\n
            "- For budget-related questions or when suggesting budget improvements, always end with: 'Would you like to use our tool to budget?'\n"
            "- For spending pattern or analysis questions, always end with: 'Would you like to deep dive into your expenses?'\n"
            "- If the user asks for a summary, breakdown, or advice about a category, merchant, or time period, use the most relevant summaries and insights provided in the context.\n"
            "- Never fabricate or guess transaction details. Only use the data and insights provided in the context.\n"
        """
        )
        
        # Always use JSON or readable lists for context, not markdown tables
        use_flattened = False
        full_context_parts = [system_prompt]
        if memory_summary:
            full_context_parts.append(f"Conversation summary so far: {memory_summary}")
        if any(word in message.lower() for word in ['summary', 'overview', 'total', 'all', 'everything']):
            try:
                full_context_parts.append(f"Customer Summary: {json.dumps(customer_summaries[customer_id])}")
            except Exception as e:
                print(f"Error in customer summary processing: {str(e)}")
                print(traceback.format_exc())
        if intent['category'] and customer_id in merchant_summaries:
            try:
                cat = intent['category']
                cat_summary = merchant_summaries[customer_id].get(cat.title()) or merchant_summaries[customer_id].get(cat.capitalize()) or merchant_summaries[customer_id].get(cat.upper()) or merchant_summaries[customer_id].get(cat.lower())
                if cat_summary:
                    full_context_parts.append(f"Top Merchant by category ({cat}): {json.dumps(cat_summary)}")
            except Exception as e:
                print(f"Error in category summary processing: {str(e)}")
                print(traceback.format_exc())
        if intent['month'] and customer_id in merchants_by_month:
            try:
                month = intent['month']
                if re.match(r'\d{4}-\d{2}', month):
                    ym = month
                else:
                    months = [
                        ('january', '01'), ('february', '02'), ('march', '03'), ('april', '04'), ('may', '05'), ('june', '06'),
                        ('july', '07'), ('august', '08'), ('september', '09'), ('october', '10'), ('november', '11'), ('december', '12'),
                        ('jan', '01'), ('feb', '02'), ('mar', '03'), ('apr', '04'), ('jun', '06'), ('jul', '07'), ('aug', '08'),
                        ('sep', '09'), ('oct', '10'), ('nov', '11'), ('dec', '12')
                    ]
                    ym = None
                    for m, mm in months:
                        if m in month:
                            ym = f"2025-{mm}"
                            break
                if ym and ym in merchants_by_month[customer_id]:
                    full_context_parts.append(f"Top Merchants by Month ({ym}): {json.dumps(merchants_by_month[customer_id][ym])}")
            except Exception as e:
                print(f"Error in month summary processing: {str(e)}")
                print(traceback.format_exc())
        # Add relevant insights to context
        if relevant_insights:
            try:
                # Add as readable list
                insights_text = '\n'.join([f"- {insight['type']} ({insight['category']}): {insight['narrative']}" for insight in relevant_insights])
                full_context_parts.append(f"Relevant Insights:\n{insights_text}")
            except Exception as e:
                print(f"Error in insights context processing: {str(e)}")
                print(traceback.format_exc())
        full_context = "\n".join(full_context_parts)
        
        # Count and print tokens for each component
        print("\n=== Token Count Breakdown ===")
        print(f"System Prompt: {count_tokens(system_prompt)} tokens")
        print(f"Memory Summary: {count_tokens(memory_summary)} tokens")
        for i, ctx in enumerate(context[1:], 1):  # Skip memory summary as it's already counted
            print(f"Context Part {i}: {count_tokens(ctx)} tokens")
        print(f"User Message: {count_tokens(message)} tokens")
        print(f"Total Context: {count_tokens(full_context)} tokens")
        print("===========================\n")
        
        # Call OpenAI for the main answer
        try:
            print("Making OpenAI API call...")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": full_context},
                    {"role": "user", "content": message}
                ],
                max_tokens=400
            )
            answer = response.choices[0].message.content
            
            # Count and print response tokens
            print(f"Response Tokens: {count_tokens(answer)} tokens")
            
            # Initialize summary
            summary = ""
            
            # Update conversation history
            try:
                history = conversation_history.get(customer_id, {}).get('history', [])
                history.append({"user": message, "bot": answer})
                history = history[-3:]  # Keep only last 3 messages
                
                # Generate a summary using OpenAI
                try:
                    summary_prompt = "Summarize the following conversation so far in 1-2 sentences for context:\n" + "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in history])
                    summary_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": summary_prompt}
                        ],
                        max_tokens=100
                    )
                    summary = summary_response.choices[0].message.content.strip()
                    print(f"Generated summary: {summary}")
                except Exception as e:
                    print(f"OpenAI summary error: {str(e)}")
                    print(traceback.format_exc())
                    summary = "Error generating summary"
                
                conversation_history[customer_id] = {"history": history, "summary": summary}
            except Exception as e:
                print(f"Error updating conversation history: {str(e)}")
                print(traceback.format_exc())
            
            # Prepare response data
            response_data = {
                "answer": answer,
                "memory": summary
            }
            # Set show_raw_transactions flag based on LLM answer (robust to phrasing and quotes)
            if 'show raw transactions' in answer.lower():
                response_data["show_raw_transactions"] = True
            else:
                response_data["show_raw_transactions"] = False
            
            # LLM-based insight selection
            context_agent = ContextAgent(client)
            if relevant_insights:
                insight_result = context_agent.select_relevant_insight(message, relevant_insights, memory_summary)
                print(f"LLM insight selection result: {insight_result}")
                if insight_result.get('selected_insight') is not None and insight_result.get('confidence', 0) > 0.7:
                    selected_insight = relevant_insights[insight_result['selected_insight']]
                    print(f"Selected insight: {selected_insight}")
                    if not answer.endswith(selected_insight['narrative']):
                        answer += f"\n\n{selected_insight['narrative']}"
                    if selected_insight['type'] in ['trend', 'anomaly', 'budget_overrun']:
                        response_data["show_budget_tool"] = True
                    elif selected_insight['type'] in ['savings', 'category_deepdive', 'deep_dive', 'category_deepdive']:
                        response_data["show_spends_analyzer"] = True
                    response_data["answer"] = answer
            
            # Also check for direct button triggers in the message
            message_lower = message.lower()
            if any(phrase in message_lower for phrase in ['would you like to use our tool to budget', 'use our tool to budget', 'show budget tool']):
                response_data["show_budget_tool"] = True
                print("Setting show_budget_tool to True based on direct message trigger")
            elif any(phrase in message_lower for phrase in ['would you like to deep dive into your expenses', 'deep dive into your expenses', 'show spends analyzer']):
                response_data["show_spends_analyzer"] = True
                print("Setting show_spends_analyzer to True based on direct message trigger")
            
            print("\n=== Final Response Data ===")
            print(f"Answer: {response_data['answer'][:100]}...")
            print(f"Show raw transactions: {response_data.get('show_raw_transactions', False)}")
            print(f"Show budget tool: {response_data.get('show_budget_tool', False)}")
            print(f"Show spends analyzer: {response_data.get('show_spends_analyzer', False)}")
            print("===========================\n")
            
            return JSONResponse(
                content=response_data,
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            print(traceback.format_exc())
            error_message = str(e)
            if "insufficient_quota" in error_message.lower():
                return JSONResponse(
                    status_code=429,
                    content={"error": "OpenAI API quota exceeded. Please check your billing details or try again later."}
                )
            elif "invalid_api_key" in error_message.lower():
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid OpenAI API key. Please check your API key configuration."}
                )
            else:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Error communicating with OpenAI API: {error_message}"}
                )
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "An error occurred while processing your request", "details": str(e)},
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )

class TransactionFilterAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.supported_fields = {
            'date': ['exact_date', 'year_month', 'relative_date', 'range'],
            'category': ['exact_match', 'contains'],
            'amount': ['exact', 'greater_than', 'less_than', 'range'],
            'merchant': ['exact_match', 'contains'],
            'transaction_type': ['exact_match'],
            'currency': ['exact_match'],
            'transaction_mode': ['exact_match']
        }
        self.field_map = {
            "merchant": "merchant_name",
            "category": "transaction_category",
            "amount": "transaction_amount",
            "date": "transaction_date",
            "type": "transaction_type",
            "currency": "transaction_currency",
            "mode": "transaction_mode",
        }

    def _normalize_date(self, date_str, txns=None):
        """
        Convert date_str like '10 mar', '10th march', '2025-03-10' to 'YYYY-MM-DD'.
        If year is missing, use the year from the first transaction or default to current year.
        """
        if not date_str:
            return None
        date_str = date_str.strip().lower()
        # Try ISO format first
        try:
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
        # Try to extract day, month, year
        m = re.match(r"(\d{1,2})(?:st|nd|rd|th)?[\s-]*([a-zA-Z]+)[\s-]*(\d{4})?", date_str)
        if m:
            day = int(m.group(1))
            month_str = m.group(2)[:3]
            year = m.group(3)
            # Try to get year from txns if not specified
            if not year and txns and len(txns) > 0:
                year = txns[0].get('transaction_date', '2025-01-01')[:4]
            if not year:
                year = str(datetime.datetime.now().year)
            month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
            month = month_map.get(month_str, 1)
            try:
                dt = datetime.date(int(year), int(month), int(day))
                return dt.strftime("%Y-%m-%d")
            except Exception:
                return None
        # Try month-year only
        m = re.match(r"([a-zA-Z]+)[\s-]*(\d{4})", date_str)
        if m:
            month_str = m.group(1)[:3]
            year = m.group(2)
            month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
            month = month_map.get(month_str, 1)
            return f"{year}-{str(month).zfill(2)}-01"
        return None

    def _preprocess_filters(self, filter_params, txns):
        """
        Normalize date values in filter_params to 'YYYY-MM-DD' if needed.
        Also normalize 'range' operator values from dict to list if needed.
        """
        for f in filter_params.get('filters', []):
            # Normalize 'range' operator value from dict to list
            if f['operator'] == 'range':
                if isinstance(f['value'], dict) and 'start_date' in f['value'] and 'end_date' in f['value']:
                    f['value'] = [f['value']['start_date'], f['value']['end_date']]
            if f['field'] == 'date':
                if f['operator'] in ['exact_date', 'greater_than', 'less_than']:
                    f['value'] = self._normalize_date(f['value'], txns)
                elif f['operator'] == 'relative_date' and isinstance(f['value'], list) and len(f['value']) == 2:
                    f['value'] = [self._normalize_date(f['value'][0], txns), self._normalize_date(f['value'][1], txns)]
                elif f['operator'] == 'range' and isinstance(f['value'], list) and len(f['value']) == 2:
                    f['value'] = [self._normalize_date(f['value'][0], txns), self._normalize_date(f['value'][1], txns)]
        return filter_params

    def parse_query(self, query: str, txns=None) -> Dict:
        """
        Parse natural language query into structured filter parameters
        """
        prompt = self._build_parsing_prompt(query)
        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
        )
        filter_params = self._parse_llm_response(response)
        return self._preprocess_filters(filter_params, txns)

    def _build_parsing_prompt(self, query: str) -> str:
        """
        Build prompt for LLM to parse query
        """
        return f"""
        Parse the following query into filter parameters:
        Query: {query}
        
        Return a JSON object with the following structure:
        {{
            "filters": [
                {{
                    "field": "field_name",
                    "operator": "operator",
                    "value": "value",
                    "confidence": 0.0-1.0
                }}
            ],
            "logic": "AND/OR",
            "ambiguities": [
                {{
                    "field": "field_name",
                    "possible_values": ["value1", "value2"],
                    "confidence": [0.0-1.0, 0.0-1.0]
                }}
            ]
        }}
        """

    def _get_system_prompt(self) -> str:
        """
        Get system prompt for LLM
        """
        return """
        You are a transaction query parsing agent. Your job is to convert natural language queries about transactions into structured filter parameters.
        You should:
        1. Identify all filter conditions in the query
        2. Map them to supported fields and operators
        3. Extract values and normalize them (dates, amounts, etc.)
        4. Identify any ambiguities in the query
        5. Provide confidence scores for your understanding

        Supported fields and operators:
        - date: exact_date (YYYY-MM-DD), year_month (YYYY-MM), relative_date (last week, last month, etc.)
        - if year not mentioned assume 2025 (current year)
        - category: exact_match, contains
        - amount: exact, greater_than, less_than, range
        - merchant: exact_match, contains
        - transaction_type: exact_match
        - currency: exact_match
        - transaction_mode: exact_match

        For dates:
        - Convert relative dates to actual dates (e.g., "last week" to date range)
        - Handle month names and abbreviations
        - Handle year-month format (YYYY-MM)

        For amounts:
        - Normalize currency amounts
        - Handle ranges (e.g., "between 100 and 200")
        - Handle relative amounts (e.g., "more than 1000")

        For categories and merchants:
        - Handle partial matches
        - Handle case-insensitive matching
        - Handle common variations in names
        """

    def _parse_llm_response(self, response) -> Dict:
        """
        Parse LLM response into structured format
        """
        try:
            # Extract JSON from response
            content = response.choices[0].message.content
            # Find JSON object in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in response")
            
            # Parse JSON
            filter_params = json.loads(json_match.group(0))
            
            # Validate structure
            if not isinstance(filter_params, dict):
                raise ValueError("Invalid filter parameters structure")
            
            if 'filters' not in filter_params:
                raise ValueError("Missing 'filters' in parameters")
            
            # Validate each filter
            for filter_item in filter_params['filters']:
                if not all(k in filter_item for k in ['field', 'operator', 'value']):
                    raise ValueError("Invalid filter item structure")
                
                if filter_item['field'] not in self.supported_fields:
                    raise ValueError(f"Unsupported field: {filter_item['field']}")
                
                if filter_item['operator'] not in self.supported_fields[filter_item['field']]:
                    raise ValueError(f"Unsupported operator for field {filter_item['field']}: {filter_item['operator']}")
            
            return filter_params
            
        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
            print(f"Raw response: {content}")
            raise

    def apply_filters(self, transactions: List[Dict], filter_params: Dict) -> List[Dict]:
        if not filter_params.get('filters'):
            return transactions
        filtered_txns = transactions
        logic = filter_params.get('logic', 'AND')
        for filter_item in filter_params['filters']:
            field = filter_item['field']
            operator = filter_item['operator']
            value = filter_item['value']
            if logic == 'AND':
                filtered_txns = [t for t in filtered_txns if self._apply_filter(t, field, operator, value)]
            else:  # OR logic
                filtered_txns = [t for t in transactions if self._apply_filter(t, field, operator, value)]
        return filtered_txns

    def _apply_filter(self, transaction: Dict, field: str, operator: str, value: any) -> bool:
        mapped_field = self.field_map.get(field, field)
        if mapped_field not in transaction:
            return False
        transaction_value = transaction[mapped_field]
        # Date handling
        if operator == 'exact_match':
            return str(transaction_value).lower() == str(value).lower()
        elif operator == 'contains':
            return str(value).lower() in str(transaction_value).lower()
        elif operator == 'exact_date':
            return transaction_value == value
        elif operator == 'year_month':
            return transaction_value.startswith(value)
        elif operator == 'relative_date':
            # Handle relative date ranges
            start_date, end_date = value
            return start_date <= transaction_value <= end_date
        elif operator == 'greater_than':
            # Support for date and amount
            try:
                if mapped_field == 'transaction_date':
                    return transaction_value > value
                return float(transaction_value) > float(value)
            except Exception:
                return False
        elif operator == 'less_than':
            try:
                if mapped_field == 'transaction_date':
                    return transaction_value < value
                return float(transaction_value) < float(value)
            except Exception:
                return False
        elif operator == 'range':
            min_val, max_val = value
            try:
                if mapped_field == 'transaction_date':
                    return min_val <= transaction_value <= max_val
                return float(min_val) <= float(transaction_value) <= float(max_val)
            except Exception:
                return False
        return False

class ContextAgent(TransactionFilterAgent):
    def select_relevant_insight(self, message, insights, context=None):
        prompt = self._build_insight_prompt(message, insights, context)
        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a finance assistant. Given a user message, a list of insights, and optional conversation context, select the most relevant insight for the user's message. Return a JSON object with selected_insight (index), reason, and confidence (0.0-1.0)."},
                {"role": "user", "content": prompt}
            ]
        )
        return self._parse_insight_response(response)

    def _build_insight_prompt(self, message, insights, context):
        return f"""
User message: {message}
Available insights (as a list):
{json.dumps(insights, indent=2)}
Conversation context: {context or ''}

Please select the most relevant insight for the user's message. Return a JSON object:
{{
  \"selected_insight\": <index>,
  \"reason\": <why this insight>,
  \"confidence\": <0.0-1.0>
}}
"""

    def _parse_insight_response(self, response):
        content = response.choices[0].message.content
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in response")
            return json.loads(json_match.group(0))
        except Exception as e:
            print(f"Error parsing LLM insight response: {str(e)}")
            print(f"Raw response: {content}")
            return {"selected_insight": None, "reason": "Parse error", "confidence": 0.0}

@app.get("/raw_transactions")
def get_raw_transactions(
    customer_id: str = Query(...),
    message: str = Query(None),
    use_flattened: bool = Query(False)
):
    try:
        # Initialize agent
        agent = TransactionFilterAgent(client)
        
        # Get transactions
        txns = raw_transactions.get(customer_id, [])
        
        if message:
            try:
                # Parse query using LLM
                filter_params = agent.parse_query(message, txns)
                
                # Apply filters
                filtered_txns = agent.apply_filters(txns, filter_params)
                
                # Handle ambiguities if any
                if filter_params.get('ambiguities'):
                    # Log ambiguities for monitoring
                    print(f"Query ambiguities: {filter_params['ambiguities']}")
                    
                    # Optionally, return ambiguities to user
                    if use_flattened:
                        return JSONResponse({
                            "transactions": flatten_to_markdown(filtered_txns, "raw_transactions"),
                            "ambiguities": filter_params['ambiguities']
                        })
                    return JSONResponse({
                        "transactions": filtered_txns,
                        "ambiguities": filter_params['ambiguities']
                    })
                
                txns = filtered_txns
                
            except Exception as e:
                print(f"Error in LLM-based filtering: {str(e)}")
                print(traceback.format_exc())
                # Fallback to original filtering if LLM-based filtering fails
                exact_date = None
                year_month = None
                category = None
                
                # Extract exact date (YYYY-MM-DD)
                m = re.search(r"(\d{4}-\d{2}-\d{2})", message)
                if m:
                    exact_date = m.group(1)
                else:
                    # Extract "12th feb 2025", "12 feb 2025", "12th feb", etc.
                    dmy = re.search(r"(\d{1,2})(?:st|nd|rd|th)?[ -]?([a-zA-Z]+)[ -]?(\d{4})?", message)
                    if dmy:
                        d = dmy.group(1)
                        mon = dmy.group(2)
                        y = dmy.group(3) or str((txns[0]['transaction_date'] if txns else '2025-01-01')[:4])
                        monthNum = str((re.match(r'jan', mon, re.I) and 1) or (re.match(r'feb', mon, re.I) and 2) or (re.match(r'mar', mon, re.I) and 3) or (re.match(r'apr', mon, re.I) and 4) or (re.match(r'may', mon, re.I) and 5) or (re.match(r'jun', mon, re.I) and 6) or (re.match(r'jul', mon, re.I) and 7) or (re.match(r'aug', mon, re.I) and 8) or (re.match(r'sep', mon, re.I) and 9) or (re.match(r'oct', mon, re.I) and 10) or (re.match(r'nov', mon, re.I) and 11) or (re.match(r'dec', mon, re.I) and 12))
                        mm = monthNum.zfill(2)
                        d = d.zfill(2)
                        exact_date = f"{y}-{mm}-{d}"
                        year_month = f"{y}-{mm}"
                
                # Extract year_month (YYYY-MM)
                if not year_month:
                    m = re.search(r"(\d{4}-\d{2})", message)
                    if m:
                        year_month = m.group(1)
                
                # Extract standalone month name
                if not year_month:
                    m = re.search(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b", message, re.I)
                    if m:
                        mon = m.group(1)
                        y = str((txns[0]['transaction_date'] if txns else '2025-01-01')[:4])
                        monthNum = str((re.match(r'jan', mon, re.I) and 1) or (re.match(r'feb', mon, re.I) and 2) or (re.match(r'mar', mon, re.I) and 3) or (re.match(r'apr', mon, re.I) and 4) or (re.match(r'may', mon, re.I) and 5) or (re.match(r'jun', mon, re.I) and 6) or (re.match(r'jul', mon, re.I) and 7) or (re.match(r'aug', mon, re.I) and 8) or (re.match(r'sep', mon, re.I) and 9) or (re.match(r'oct', mon, re.I) and 10) or (re.match(r'nov', mon, re.I) and 11) or (re.match(r'dec', mon, re.I) and 12))
                        mm = monthNum.zfill(2)
                        year_month = f"{y}-{mm}"
                
                # Extract category
                m = re.search(r'groceries|entertainment|travel|utility|food|shopping|medical|salary|offline food|online food|offline shopping|offline|online', message, re.I)
                if m:
                    category = m.group(0)
                
                if exact_date:
                    txns = [t for t in txns if t.get('transaction_date', '') == exact_date]
                elif year_month:
                    txns = [t for t in txns if t.get('transaction_date', '').startswith(year_month)]
                if category:
                    txns = [t for t in txns if category.lower() in t.get('transaction_category', '').lower()]
        
        if use_flattened:
            return JSONResponse({"transactions": flatten_to_markdown(txns, "raw_transactions")})
        return JSONResponse({"transactions": txns})
        
    except Exception as e:
        print(f"Error in raw transactions: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "An error occurred while processing your request"}
        )

@app.get("/budget_tool")
async def budget_tool(
    customer_id: str = Query(...),
    use_flattened: bool = Query(False)
):
    try:
        if customer_id not in customer_summaries:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid customer ID: {customer_id}"}
            )
        
        # Get current month's data
        current_month = max(customer_summaries[customer_id]['monthly_summary'].keys())
        monthly_data = customer_summaries[customer_id]['monthly_summary'][current_month]
        expense_details = customer_summaries[customer_id]['expense_details']
        
        # Prepare budget tool data
        budget_data = {
            "month": current_month,
            "income": monthly_data['income'],
            "expenses": monthly_data['expense'],
            "savings": monthly_data['savings_surplus_deficit'],
            "categories": []
        }
        
        # Add category-wise breakdown
        for category, details in expense_details.items():
            monthly_amount = details['monthly_breakup'].get(current_month, 0)
            if monthly_amount > 0:  # Only include categories with spending
                budget_data["categories"].append({
                    "category": category,
                    "amount": monthly_amount,
                    "pct_of_income": details['pct_of_income'],
                    "monthly_avg": details['monthly_avg']
                })
        
        if use_flattened:
            # Create markdown table
            md = f"### Budget Overview for {current_month}\n\n"
            md += "| Metric | Amount (AED) |\n"
            md += "|--------|-------------|\n"
            md += f"| Income | {budget_data['income']} |\n"
            md += f"| Expenses | {budget_data['expenses']} |\n"
            md += f"| Savings | {budget_data['savings']} |\n\n"
            
            md += "### Category Breakdown\n\n"
            md += "| Category | Current Month | % of Income | Monthly Average |\n"
            md += "|----------|---------------|-------------|-----------------|\n"
            for cat in budget_data["categories"]:
                md += f"| {cat['category']} | {cat['amount']} | {cat['pct_of_income']}% | {cat['monthly_avg']} |\n"
            
            return JSONResponse({"budget": md})
        
        return JSONResponse({"budget": budget_data})
        
    except Exception as e:
        print(f"Error in budget tool: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "An error occurred while generating budget tool data"}
        )

@app.get("/spends_analyzer")
async def spends_analyzer(
    customer_id: str = Query(...),
    use_flattened: bool = Query(False)
):
    try:
        if customer_id not in customer_summaries:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid customer ID: {customer_id}"}
            )
        
        # Get all data
        summary = customer_summaries[customer_id]['summary']
        monthly_summary = customer_summaries[customer_id]['monthly_summary']
        expense_details = customer_summaries[customer_id]['expense_details']
        
        # Prepare spends analyzer data
        analyzer_data = {
            "overall": {
                "total_income": summary['total_income'],
                "total_expense": summary['total_expense'],
                "savings": summary['savings_surplus_deficit'],
                "expense_pct": summary['total_expense_pct_of_income'],
                "savings_pct": summary['savings_pct_of_income']
            },
            "monthly_trends": [],
            "category_analysis": []
        }
        
        # Add monthly trends
        for month, data in monthly_summary.items():
            analyzer_data["monthly_trends"].append({
                "month": month,
                "income": data['income'],
                "expense": data['expense'],
                "savings": data['savings_surplus_deficit']
            })
        
        # Add category analysis
        for category, details in expense_details.items():
            if details['total'] > 0:  # Only include categories with spending
                analyzer_data["category_analysis"].append({
                    "category": category,
                    "total": details['total'],
                    "pct_of_income": details['pct_of_income'],
                    "monthly_avg": details['monthly_avg'],
                    "monthly_breakup": details['monthly_breakup']
                })
        
        if use_flattened:
            # Create markdown tables
            md = "### Overall Financial Summary\n\n"
            md += "| Metric | Amount (AED) |\n"
            md += "|--------|-------------|\n"
            md += f"| Total Income | {analyzer_data['overall']['total_income']} |\n"
            md += f"| Total Expenses | {analyzer_data['overall']['total_expense']} |\n"
            md += f"| Total Savings | {analyzer_data['overall']['savings']} |\n"
            md += f"| Expenses % of Income | {analyzer_data['overall']['expense_pct']}% |\n"
            md += f"| Savings % of Income | {analyzer_data['overall']['savings_pct']}% |\n\n"
            
            md += "### Monthly Trends\n\n"
            md += "| Month | Income | Expenses | Savings |\n"
            md += "|-------|---------|-----------|----------|\n"
            for trend in analyzer_data["monthly_trends"]:
                md += f"| {trend['month']} | {trend['income']} | {trend['expense']} | {trend['savings']} |\n"
            
            md += "\n### Category Analysis\n\n"
            md += "| Category | Total Spend | % of Income | Monthly Average |\n"
            md += "|----------|-------------|-------------|-----------------|\n"
            for cat in analyzer_data["category_analysis"]:
                md += f"| {cat['category']} | {cat['total']} | {cat['pct_of_income']}% | {cat['monthly_avg']} |\n"
            
            return JSONResponse({"analysis": md})
        
        return JSONResponse({"analysis": analyzer_data})
        
    except Exception as e:
        print(f"Error in spends analyzer: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "An error occurred while generating spends analysis"}
        )

@app.get("/health")
def health():
    return {"status": "ok"} 