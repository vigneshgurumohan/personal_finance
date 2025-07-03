import traceback, json, os, re
import openai
from fastapi import Form, Query, HTTPException
from pathlib import Path
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse, JSONResponse
from api.utils.agent import TransactionFilterAgent, ContextAgent
from api.utils.helper import flatten_to_markdown, count_tokens, detect_intent, extract_keywords, find_relevant_insights
from api.utils.modals import ChatRequest, ChatResponse, PersonalFinanceAssistant 

# Load environment variables from .env file
load_dotenv('.env')

# Validate OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv("MODEL")

print(f"API Key loaded: {api_key}")  # Print first and last 10 chars for verification

# In-memory conversation history per customer_id (for model-based summary)
conversation_history = {}

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your env.txt file.")

client = openai.OpenAI(api_key=api_key)

# Test OpenAI connection
try:
    test_response = client.chat.completions.create(
        model=os.getenv("MODEL", "gpt-3.5-turbo"),
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    print("OpenAI API connection successful")
except Exception as e:
    print(f"OpenAI API connection failed: {str(e)}")
    print(traceback.format_exc())
    raise

BASE_DIR = Path(__file__).resolve().parent.parent.parent

try:
    with open(BASE_DIR / 'summaries' / 'raw_transactions_by_customer.json') as f:
        raw_transactions = json.load(f)
    
    with open(BASE_DIR / 'summaries' / 'customer_transaction_summaries.json') as f:
        customer_summaries = json.load(f)
    
    with open(BASE_DIR / 'summaries' / 'top_merchants_by_category.json') as f:
        merchant_summaries = json.load(f)
    
    with open(BASE_DIR / 'summaries' / 'top_merchants_by_month.json') as f:
        merchants_by_month = json.load(f)
    
    with open(BASE_DIR / 'summaries' / 'insights_summary.json') as f:
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

def spends_analyzer_service(
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

def budget_tool_services(
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

def get_raw_transactions_service(
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

def chat_old_service(
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
                model=os.getenv("MODEL", "gpt-3.5-turbo"),
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
                        model=os.getenv("MODEL", "gpt-3.5-turbo"),
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

def chat_service(request: ChatRequest):
    """Main chat endpoint"""
    assistant = PersonalFinanceAssistant(api_key,model)
    try:
        response, tool_used, data, chart, destination, itinerary = assistant.chat(
            request.message, 
            request.customer_id, 
            request.session_id
        )
        
        # Backend logging to track what's being sent
        print(f"\n🔍 BACKEND RESPONSE LOGGING:")
        print(f"📝 Message: {request.message}")
        print(f"👤 Customer: {request.customer_id}")
        print(f"🔧 Tool Used: {tool_used}")
        print(f"💬 Response Length: {len(response) if response else 0} chars")
        print(f"📊 Data Records: {len(data) if data else 0}")
        print(f"📈 Chart Available: {bool(chart and chart.get('chart_type'))}")
        
        if data:
            print(f"📋 Data Sample: {data[0] if data else 'None'}")
        
        if chart:
            print(f"🎯 Chart Type: {chart.get('chart_type', 'None')}")
            print(f"🎯 Chart Keys: {list(chart.keys()) if chart else []}")
        
        print(f"🔚 End Backend Logging\n")
        
        return ChatResponse(
            response=response,
            tool_used=tool_used,
            session_id=request.session_id,
            data=data,
            chart=chart,
            destination=destination,
            itinerary=itinerary
        )
    except Exception as e:
        print(f"❌ BACKEND ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
