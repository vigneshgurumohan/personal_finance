# 🏦 Personal Finance Assistant

A comprehensive AI-powered personal finance analysis and chat application that provides intelligent insights into your spending patterns, budget recommendations, and financial trends using OpenAI's GPT models.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Data Processing](#-data-processing)
- [Security](#-security)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## ✨ Features

### 🤖 AI-Powered Chat Interface
- **Natural Language Queries**: Ask questions about your finances in plain English
- **Context-Aware Responses**: AI remembers conversation context and provides personalized insights
- **Multi-Modal Analysis**: Supports both structured data and natural language responses
- **Real-time Processing**: Instant responses with intelligent data interpretation

### 📊 Comprehensive Financial Analysis
- **Transaction Summaries**: Detailed breakdowns by customer, category, and time period
- **Merchant Analysis**: Top merchants by category and monthly spending patterns
- **Trend Detection**: Identifies spending trends and anomalies
- **Budget Insights**: Income vs. expense analysis with actionable recommendations

### 🎯 Intelligent Insights Engine
- **Trend Analysis**: Compares current spending with historical averages
- **Anomaly Detection**: Identifies unusual transactions and spending patterns
- **Savings Opportunities**: Highlights areas where spending has decreased
- **Recurring Payment Tracking**: Monitors regular payments and subscriptions
- **Budget Overrun Alerts**: Warns when expenses exceed income

### 🛠️ Advanced Tools
- **Raw Transaction Filtering**: Advanced query parsing for specific transaction searches
- **Budget Planning Tool**: Interactive budget creation and analysis
- **Spends Analyzer**: Deep dive into expense patterns and categories
- **Markdown Table Generation**: Clean, formatted data presentation

### 🔄 Data Processing Pipeline
- **Automated Summaries**: Generates comprehensive financial summaries
- **Multi-format Support**: Handles CSV data with flexible parsing
- **Real-time Updates**: Processes new transactions and updates insights
- **Scalable Architecture**: Supports multiple customers with isolated data

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   Data Layer    │
│   (HTML/JS)     │◄──►│   Backend       │◄──►│   (CSV/JSON)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   OpenAI API    │
                       │   (GPT-3.5)     │
                       └─────────────────┘
```

### Core Components

1. **FastAPI Web Server** (`main.py`)
   - RESTful API endpoints
   - WebSocket support for real-time chat
   - Static file serving
   - CORS middleware

2. **Data Processing Engine** (`all_summaries.py`)
   - Transaction data analysis
   - Summary generation
   - Insight calculation
   - JSON output formatting

3. **AI Integration Layer**
   - OpenAI GPT-3.5-turbo integration
   - Natural language processing
   - Context management
   - Response generation

4. **Frontend Interface** (`templates/index.html`)
   - Interactive chat interface
   - Real-time message display
   - Markdown rendering
   - Dynamic button controls

## 📋 Prerequisites

- **Python 3.8+**
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **Transaction Data** in CSV format
- **Internet Connection** for OpenAI API access

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd personal_finance
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the root directory:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Additional Configuration
# DATABASE_URL=your_database_url_here
# SECRET_KEY=your_secret_key_here
# DEBUG=True
```

### 4. Data Preparation

Place your transaction data in `data/raw_transactions.csv` with the following columns:

```csv
customer_id,transaction_date,transaction_type,transaction_category,merchant_name,transaction_amount,transaction_currency,transaction_mode
1234,01-01-2024,Income,Salary,Company ABC,5000,AED,Bank Transfer
1234,02-01-2024,Expense,Groceries,Supermarket XYZ,150,AED,Credit Card
```

### 5. Generate Summaries

```bash
python all_summaries.py
```

This will create the following files in the `summaries/` directory:
- `customer_transaction_summaries.json`
- `top_merchants_by_category.json`
- `top_merchants_by_month.json`
- `raw_transactions_by_customer.json`
- `insights_summary.json`

### 6. Start the Application

```bash
python main.py
```

The application will be available at `http://localhost:8000`

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes | - |
| `DEBUG` | Enable debug mode | No | False |

### Insight Configuration

Modify `all_summaries.py` to customize insight generation:

```python
INSIGHT_CONFIG = {
    'comparison_months': 6,  # Months for trend comparison
    'recurring_months': 6,   # Months to check for recurring payments
    'insight_types': {
        'current': ['trend', 'anomaly', 'savings'],
        'overall': ['recurring', 'budget_overrun', 'deep_dive']
    },
    'insights_expenses_only': True  # Only analyze expenses
}
```

## 💬 Usage

### Web Interface

1. **Open the Application**: Navigate to `http://localhost:8000`
2. **Enter Customer ID**: Input your customer identifier
3. **Start Chatting**: Ask questions about your finances

### Example Queries

```
"What is my total income and expenses?"
"Show me my spending on groceries last month"
"Which merchants do I spend the most with?"
"What are my biggest expense categories?"
"Show me transactions above 500 AED"
"Compare my spending this month vs last month"
```

### Advanced Features

#### Raw Transaction Filtering
- **Natural Language Queries**: "Show transactions from Starbucks in January"
- **Complex Filters**: "Expenses above 1000 AED in the last 3 months"
- **Date Ranges**: "Transactions between January 1st and March 31st"

#### Budget Tool
- **Interactive Budgeting**: Create and track budget categories
- **Spending Analysis**: Compare actual vs. budgeted amounts
- **Recommendations**: AI-powered budget suggestions

#### Spends Analyzer
- **Deep Dive Analysis**: Detailed category breakdowns
- **Trend Visualization**: Spending patterns over time
- **Merchant Insights**: Top spending locations and patterns

## 🔌 API Endpoints

### Core Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/` | GET | Main chat interface | - |
| `/chat` | POST | Process chat messages | `customer_id`, `message`, `use_flattened` |
| `/raw_transactions` | GET | Filter raw transactions | `customer_id`, `message`, `use_flattened` |
| `/budget_tool` | GET | Budget analysis tool | `customer_id`, `use_flattened` |
| `/spends_analyzer` | GET | Detailed spending analysis | `customer_id`, `use_flattened` |
| `/health` | GET | Health check endpoint | - |

### Request Examples

#### Chat Endpoint
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "customer_id=1234&message=What is my total spending?"
```

#### Raw Transactions
```bash
curl "http://localhost:8000/raw_transactions?customer_id=1234&message=groceries"
```

### Response Format

```json
{
  "answer": "Your total spending on groceries is 1,250 AED...",
  "memory": "Previous conversation context...",
  "transactions": [...],
  "insights": [...]
}
```

## 📊 Data Processing

### Input Data Format

The application expects CSV data with the following structure:

```csv
customer_id,transaction_date,transaction_type,transaction_category,merchant_name,transaction_amount,transaction_currency,transaction_mode
```

### Generated Summaries

1. **Customer Transaction Summaries**
   - Total income/expenses by customer
   - Monthly breakdowns
   - Category-wise analysis
   - Savings/surplus calculations

2. **Merchant Analysis**
   - Top merchants by category
   - Monthly merchant rankings
   - Spending percentages and trends

3. **Insights Summary**
   - Trend analysis
   - Anomaly detection
   - Budget recommendations
   - Actionable insights

### Data Flow

```
Raw CSV → Data Processing → JSON Summaries → AI Analysis → User Interface
```

## 🔒 Security

### API Key Management

- **Environment Variables**: API keys stored in `.env` files
- **Git Ignore**: `.env` files excluded from version control
- **No Hardcoding**: No API keys in source code
- **Secure Loading**: Keys loaded at runtime only

### Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for all sensitive data
3. **Regular key rotation** for production environments
4. **Monitor API usage** to prevent quota exhaustion

### Production Deployment

```bash
# Set production environment variables
export OPENAI_API_KEY=your_production_key
export DEBUG=False

# Run with production server
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🐛 Troubleshooting

### Common Issues

#### 1. OpenAI API Connection Failed
```
Error: OpenAI API connection failed
```
**Solution**: Verify your API key in the `.env` file

#### 2. Missing Summary Files
```
Error: Error loading summary files
```
**Solution**: Run `python all_summaries.py` to generate summaries

#### 3. Invalid Customer ID
```
Error: Customer not found
```
**Solution**: Check that the customer ID exists in your data

#### 4. Data Format Issues
```
Error: Invalid CSV format
```
**Solution**: Ensure your CSV has the required columns

### Debug Mode

Enable debug mode for detailed error information:

```env
DEBUG=True
```

### Logging

The application provides detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests** for new functionality
5. **Submit a pull request**

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions
- Include error handling

### Testing

```bash
# Run basic tests
python -m pytest tests/

# Test specific components
python test_chat.py
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for providing the GPT API
- **FastAPI** for the web framework
- **Pandas** for data processing
- **Community contributors** for feedback and improvements

## 📞 Support

For support and questions:

1. **Check the documentation** above
2. **Review the troubleshooting section**
3. **Open an issue** on GitHub
4. **Contact the maintainers**

---

**Note**: This application is designed for personal finance analysis and should not be used for financial advice. Always consult with qualified financial professionals for important financial decisions. 