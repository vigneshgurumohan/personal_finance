import tiktoken
from typing import List, Dict, Optional

# Initialize tokenizer
import os
tokenizer = tiktoken.encoding_for_model(os.getenv("MODEL", "gpt-3.5-turbo"))

# Helper to extract topic from user message
merchant_keywords = set()
category_keywords = set()
# Build sets from data
# for cust in merchant_summaries.values():
#     for cat, merchants in cust.items():
#         category_keywords.add(cat.lower())
#         for m in merchants:
#             merchant_keywords.add(m['merchant_name'].lower())

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
