import pandas as pd
import json
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

# Configuration for insight generation
INSIGHT_CONFIG = {
    'comparison_months': 6,  # Number of months to use for averages
    'recurring_months': 6,   # Number of months to check for recurring
    'insight_types': {       # Which insights to generate
        'current': ['trend', 'anomaly', 'savings'],
        'overall': ['recurring', 'budget_overrun', 'deep_dive']
    },
    'insights_expenses_only': True  # Only analyze 'Expense' transactions for insights
}

# Validate OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

client = openai.OpenAI(api_key=api_key)

def generate_narrative(insight):
    try:
        prompt = (
            f"Rewrite this insight for a customer in a concise, factual, and neutral way. All values are in AED currency. Do NOT use conversational openers, greetings, or phrases like 'Hey there!', 'Fun fact:', 'Just a heads up!', etc. Only state the data and analysis, so the narrative can be appended or woven into any response seamlessly. Include the actual values in the message.\n"
            f"Insight: {insight['message']}\n"
            f"Data: {insight.get('data', {})}"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=80
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Warning: Could not generate narrative for insight due to: {str(e)}")
        return insight['message']  # Fallback to original message

def generate_overall_narrative(insight_narratives):
    try:
        prompt = (
            "Given these customer insights, generate a concise, friendly summary of the most important things the customer should know this month.Do NOT use conversational openers, greetings, or phrases like 'Hey there!', 'Fun fact:', 'Just a heads up!', etc. Only state the data and analysis, so the narrative can be appended or woven into any response seamlessly.  "
            "Highlight trends, anomalies, and actionable items.All values are in AED currency.\n"
            f"Insights:\n- " + "\n- ".join(insight_narratives)
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=120
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Warning: Could not generate overall narrative due to: {str(e)}")
        return "Summary of insights: " + "; ".join(insight_narratives)  # Fallback to simple concatenation

# Ensure output directory exists
os.makedirs('summaries', exist_ok=True)

# Read the transaction data
file_path = 'data/raw_transactions.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# --- 1. Raw transactions by customer ---
df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%d-%m-%Y')
df['year_month'] = df['transaction_date'].dt.strftime('%Y-%m')
raw_by_customer = {}
for customer_id, group in df.groupby('customer_id'):
    group_cpy = group.copy()
    group_cpy['transaction_date'] = group_cpy['transaction_date'].dt.strftime('%Y-%m-%d')
    raw_by_customer[str(customer_id)] = group_cpy.to_dict(orient='records')
with open('summaries/raw_transactions_by_customer.json', 'w') as f:
    json.dump(raw_by_customer, f, indent=2)

# --- 2. Customer transaction summaries ---
customer_summaries = {}
for customer_id, group in df.groupby('customer_id'):
    summary = {}
    income = round(group[group['transaction_type'] == 'Income']['transaction_amount'].sum(), 1)
    expense = round(group[group['transaction_type'] == 'Expense']['transaction_amount'].sum(), 1)
    net = round(income - expense, 1)
    summary['summary'] = {
        'total_income': income,
        'total_expense': expense,
        'savings_surplus_deficit': net,
        'total_expense_pct_of_income': round((expense / income * 100), 1) if income else None,
        'savings_pct_of_income': round((net / income * 100), 1) if income else None
    }
    # Monthly breakdown
    monthly = {}
    for ym, mgroup in group.groupby('year_month'):
        mincome = round(mgroup[mgroup['transaction_type'] == 'Income']['transaction_amount'].sum(), 1)
        mexpense = round(mgroup[mgroup['transaction_type'] == 'Expense']['transaction_amount'].sum(), 1)
        mnet = round(mincome - mexpense, 1)
        monthly[ym] = {
            'income': mincome,
            'expense': mexpense,
            'savings_surplus_deficit': mnet
        }
    summary['monthly_summary'] = monthly
    # Get all months for this customer
    all_months = sorted(group['year_month'].unique())
    # Expense details by category
    expense_details = {}
    exp_group = group[group['transaction_type'] == 'Expense']
    for cat, cat_group in exp_group.groupby('transaction_category'):
        total = round(cat_group['transaction_amount'].sum(), 1)
        pct_income = round((total / income * 100), 1) if income else None
        monthly_avg = round(total / len(monthly), 1) if monthly else 0
        monthly_breakup_raw = cat_group.groupby('year_month')['transaction_amount'].sum().to_dict()
        monthly_breakup = {m: round(float(monthly_breakup_raw.get(m, 0)), 1) for m in all_months}
        expense_details[cat] = {
            'total': total,
            'pct_of_income': pct_income,
            'monthly_avg': monthly_avg,
            'monthly_breakup': monthly_breakup
        }
    summary['expense_details'] = expense_details
    customer_summaries[str(customer_id)] = summary
with open('summaries/customer_transaction_summaries.json', 'w') as f:
    json.dump(customer_summaries, f, indent=2)

# --- 3. Top merchants by category (with rank) ---
merchant_summaries = {}
for customer_id, group in df.groupby('customer_id'):
    cust_summary = {}
    total_spends = group[group['transaction_type'] == 'Expense']['transaction_amount'].sum()
    all_months = sorted(group['year_month'].unique())
    for cat, cat_group in group[group['transaction_type'] == 'Expense'].groupby('transaction_category'):
        cat_total = cat_group['transaction_amount'].sum()
        merchant_stats = []
        for merch, merch_group in cat_group.groupby('merchant_name'):
            merch_total = merch_group['transaction_amount'].sum()
            pct_total = round((merch_total / total_spends * 100), 1) if total_spends else 0
            pct_cat = round((merch_total / cat_total * 100), 1) if cat_total else 0
            monthly_spend_raw = merch_group.groupby('year_month')['transaction_amount'].sum().to_dict()
            monthly_spend = {m: round(float(monthly_spend_raw.get(m, 0)), 1) for m in all_months}
            avg_monthly_spend = round(merch_total / len(all_months), 1) if all_months else 0
            merchant_stats.append({
                'merchant_name': merch,
                'total_spend': round(merch_total, 1),
                'pct_of_total_spends': pct_total,
                'pct_of_category_spends': pct_cat,
                'monthly_spend': monthly_spend,
                'avg_monthly_spend': avg_monthly_spend
            })
        # Sort merchants by spend descending and add rank
        merchant_stats = sorted(merchant_stats, key=lambda x: x['total_spend'], reverse=True)
        for idx, stat in enumerate(merchant_stats, 1):
            stat['rank'] = idx
        cust_summary[cat] = merchant_stats
    merchant_summaries[str(customer_id)] = cust_summary
with open('summaries/top_merchants_by_category.json', 'w') as f:
    json.dump(merchant_summaries, f, indent=2)

# --- 4. Top merchants by month ---
top_merchants_by_month = {}
for customer_id, group in df.groupby('customer_id'):
    cust_months = {}
    for ym, ym_group in group[group['transaction_type'] == 'Expense'].groupby('year_month'):
        total_monthly_spend = ym_group['transaction_amount'].sum()
        merchant_sums = ym_group.groupby(['merchant_name', 'transaction_category'])['transaction_amount'].sum().sort_values(ascending=False)
        merchant_list = []
        for idx, ((m, cat), s) in enumerate(merchant_sums.items(), 1):
            pct = round((s / total_monthly_spend * 100), 1) if total_monthly_spend else 0
            merchant_list.append({
                'merchant_name': m,
                'category': cat,
                'total_spend': round(float(s), 1),
                'pct_of_total_monthly_spend': pct,
                'rank': idx
            })
        cust_months[ym] = merchant_list
    top_merchants_by_month[str(customer_id)] = cust_months
with open('summaries/top_merchants_by_month.json', 'w') as f:
    json.dump(top_merchants_by_month, f, indent=2)

# --- Insights Generator ---
insights_summary = {}
for customer_id, group in df.groupby('customer_id'):
    insights = []
    
    # Prepare data and identify latest month
    group['year_month'] = group['transaction_date'].dt.strftime('%Y-%m')
    latest_month = group['year_month'].max()
    
    # Optionally filter only expenses for insights
    if INSIGHT_CONFIG.get('insights_expenses_only', False):
        group = group[group['transaction_type'] == 'Expense']
    
    # Get comparison period data
    all_months = sorted(group['year_month'].unique())
    comparison_months = all_months[-INSIGHT_CONFIG['comparison_months']-1:-1] if len(all_months) > INSIGHT_CONFIG['comparison_months'] else all_months[:-1]
    
    # Filter data for latest month and comparison period
    latest_data = group[group['year_month'] == latest_month]
    comparison_data = group[group['year_month'].isin(comparison_months)]
    
    # Debug: Print counts for diagnosis
    print(f"Customer {customer_id}: Latest month {latest_month} has {len(latest_data)} expense transactions.")
    print(f"Customer {customer_id}: Comparison period months: {comparison_months}, total transactions: {len(comparison_data)}.")
    
    # Generate current insights only for latest month
    if 'trend' in INSIGHT_CONFIG['insight_types']['current']:
        # Trend: Compare latest month with comparison period average
        for cat in latest_data['transaction_category'].unique():
            cat_monthly = comparison_data[comparison_data['transaction_category'] == cat].groupby('year_month')['transaction_amount'].sum()
            if len(cat_monthly) > 0:
                avg_spend = cat_monthly.mean()
                curr_spend = latest_data[latest_data['transaction_category'] == cat]['transaction_amount'].sum()
                if avg_spend > 0:
                    change_pct = round((curr_spend - avg_spend) / avg_spend * 100, 1)
                    if abs(change_pct) >= 20:
                        insights.append({
                            'id': f'trend_{cat}_{latest_month}',
                            'type': 'trend',
                            'category': cat,
                            'period': latest_month,
                            'message': f"Your spending on {cat} changed by {change_pct}% in {latest_month} compared to the {len(comparison_months)}-month average.",
                            'data': {'current': curr_spend, 'average': avg_spend, 'change_pct': change_pct},
                            'actionable': True,
                            'group': 'current'
                        })
    
    if 'anomaly' in INSIGHT_CONFIG['insight_types']['current']:
        # Anomaly: Large single transaction in latest month
        exp_group = latest_data.copy()
        for cat in exp_group['transaction_category'].unique():
            cat_txns = exp_group[exp_group['transaction_category'] == cat]
            if len(cat_txns) >= 3:
                mean = cat_txns['transaction_amount'].mean()
                std = cat_txns['transaction_amount'].std()
                threshold = mean + 2 * std
                anomalies = cat_txns[cat_txns['transaction_amount'] > threshold]
                for _, row in anomalies.iterrows():
                    insights.append({
                        'id': f'anomaly_{cat}_{row.transaction_date}_{row.merchant_name}',
                        'type': 'anomaly',
                        'category': cat,
                        'period': latest_month,
                        'message': f"Unusually high expense of {row['transaction_amount']} AED at {row['merchant_name']} in {cat} on {row['transaction_date'].strftime('%Y-%m-%d')}",
                        'data': {'amount': row['transaction_amount'], 'mean': mean, 'std': std},
                        'actionable': True,
                        'group': 'current'
                    })
    
    if 'savings' in INSIGHT_CONFIG['insight_types']['current']:
        # Savings: Category spend significantly lower than comparison period average
        for cat in latest_data['transaction_category'].unique():
            cat_monthly = comparison_data[comparison_data['transaction_category'] == cat].groupby('year_month')['transaction_amount'].sum()
            if len(cat_monthly) > 0:
                avg_spend = cat_monthly.mean()
                curr_spend = latest_data[latest_data['transaction_category'] == cat]['transaction_amount'].sum()
                if avg_spend > 0:
                    change_pct = round((curr_spend - avg_spend) / avg_spend * 100, 1)
                    if change_pct <= -20:
                        insights.append({
                            'id': f'savings_{cat}_{latest_month}',
                            'type': 'savings',
                            'category': cat,
                            'period': latest_month,
                            'message': f"You spent {abs(change_pct)}% less on {cat} in {latest_month} compared to the {len(comparison_months)}-month average.",
                            'data': {'current': curr_spend, 'average': avg_spend, 'change_pct': change_pct},
                            'actionable': True,
                            'group': 'current'
                        })
    
    if 'recurring' in INSIGHT_CONFIG['insight_types']['overall']:
        # Recurring: Merchant appears in recent months
        recent_months = all_months[-INSIGHT_CONFIG['recurring_months']:] if len(all_months) >= INSIGHT_CONFIG['recurring_months'] else all_months
        recent_data = group[group['year_month'].isin(recent_months)]
        for merch in recent_data['merchant_name'].unique():
            merchant_months = recent_data[recent_data['merchant_name'] == merch]['year_month'].sort_values().unique()
            if len(merchant_months) >= 2:
                insights.append({
                    'id': f'recurring_{merch}',
                    'type': 'recurring',
                    'merchant': merch,
                    'periods': list(merchant_months),
                    'message': f"Recurring payments to {merch} in {', '.join(merchant_months)}.",
                    'data': {'months': list(merchant_months)},
                    'actionable': False,
                    'group': 'overall'
                })
    
    if 'budget_overrun' in INSIGHT_CONFIG['insight_types']['overall']:
        # Budget Overrun: Expenses > income in latest month
        # Only makes sense if not filtering to only expenses
        if not INSIGHT_CONFIG.get('insights_expenses_only', False):
            latest_income = latest_data[latest_data['transaction_type'] == 'Income']['transaction_amount'].sum()
            latest_expense = latest_data[latest_data['transaction_type'] == 'Expense']['transaction_amount'].sum()
            if latest_expense > latest_income:
                insights.append({
                    'id': f'budgetoverrun_{latest_month}',
                    'type': 'budget_overrun',
                    'period': latest_month,
                    'message': f"Your expenses ({latest_expense} AED) exceeded your income ({latest_income} AED) in {latest_month}.",
                    'data': {'income': latest_income, 'expense': latest_expense},
                    'actionable': True,
                    'group': 'overall'
                })
    
    if 'deep_dive' in INSIGHT_CONFIG['insight_types']['overall']:
        # Category Deep Dive: Largest expense category in latest month
        if not latest_data.empty:
            cat_sums = latest_data.groupby('transaction_category')['transaction_amount'].sum()
            top_cat = cat_sums.idxmax()
            top_amt = cat_sums.max()
            insights.append({
                'id': f'deepdive_{latest_month}_{top_cat}',
                'type': 'category_deepdive',
                'category': top_cat,
                'period': latest_month,
                'message': f"Your largest expense category in {latest_month} was {top_cat} ({top_amt} AED).",
                'data': {'amount': top_amt},
                'actionable': True,
                'group': 'overall'
            })
    
    # Generate LLM narratives for each insight
    for insight in insights:
        insight['narrative'] = generate_narrative(insight)
        # Append CTA based on type
        if insight['type'] in ['trend', 'anomaly', 'budget_overrun']:
            insight['narrative'] += ' Would you like to use our tool to budget?'
        elif insight['type'] in ['savings', 'category_deepdive', 'deep_dive']:
            insight['narrative'] += ' Would you like to deep dive into your expenses?'
    
    # Generate overall narrative
    all_narratives = [ins['narrative'] for ins in insights]
    overall_narrative = generate_overall_narrative(all_narratives) if all_narratives else "No major insights for this period."
    
    insights_summary[str(customer_id)] = {
        "insights": insights,
        "overall_narrative": overall_narrative
    }
with open('summaries/insights_summary.json', 'w') as f:
    json.dump(insights_summary, f, indent=2)

print('All summaries generated in the summaries folder.') 