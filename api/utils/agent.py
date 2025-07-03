import json, re, datetime
from typing import List, Dict, Optional

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
            model=os.getenv("MODEL", "gpt-3.5-turbo"),
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
            model=os.getenv("MODEL", "gpt-3.5-turbo"),
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
