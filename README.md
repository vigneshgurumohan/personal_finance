# Personal Finance Assistant with AI

A comprehensive personal finance management system that uses AI to analyze transaction data, provide insights, and offer personalized financial advice. The system includes a web interface, REST API, and advanced analytics capabilities.

## 🚀 Features

- **AI-Powered Financial Analysis**: Uses OpenAI GPT models to analyze transaction patterns
- **Interactive Chat Interface**: Natural language queries about your finances
- **Data Visualization**: Charts and graphs for spending analysis
- **Transaction Filtering**: Advanced filtering and search capabilities
- **Budget Tools**: Monthly budget analysis and recommendations
- **Insight Generation**: Automated financial insights and trends
- **Multi-Customer Support**: Handles multiple customer profiles
- **REST API**: Full API for integration with other systems

## 📋 Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- OpenAI API key
- pip (Python package manager)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Git1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory with the following variables:
   ```env
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   MODEL=gpt-3.5-turbo
   
   # Database Configuration
   DB_USER=your_database_username
   DB_PASSWORD=your_database_password
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=your_database_name
   
   # Optional Configuration
   DEBUG=True
   LOG_LEVEL=INFO
   ```

4. **Set up the database**
   - Create a PostgreSQL database
   - Import your transaction data into the `raw_transactions` table
   - The table should have the following schema:
     ```sql
     CREATE TABLE raw_transactions (
         transaction_id VARCHAR(255) PRIMARY KEY,
         customer_id VARCHAR(255),
         merchant_name VARCHAR(255),
         transaction_date DATE,
         transaction_month VARCHAR(7),
         transaction_amount DECIMAL(10,2),
         transaction_type VARCHAR(50),
         merchant_category VARCHAR(255),
         is_online BOOLEAN,
         merchant_city VARCHAR(255),
         merchant_logo TEXT,
         transaction_currency VARCHAR(10)
     );
     ```

5. **Generate data summaries**
   ```bash
   python all_summaries.py
   ```

## 🏃‍♂️ Running the Application

### Start the FastAPI server
```bash
python main.py
```

The application will be available at `http://localhost:8000`

### Alternative: Use uvicorn directly
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📁 Project Structure

```
Git1/
├── api/                          # API layer
│   ├── routes/                   # API route definitions
│   │   ├── app_routes.py        # Main application routes
│   │   ├── health.py            # Health check endpoints
│   │   └── index_html.py        # HTML serving routes
│   ├── services/                # Business logic services
│   │   └── app_services.py      # Core application services
│   └── utils/                   # Utility modules
│       ├── agent.py             # AI agent implementations
│       ├── helper.py            # Helper functions
│       └── modals.py            # Pydantic models
├── data/                        # Data files
│   └── raw_transactions.csv     # Sample transaction data
├── summaries/                   # Generated data summaries
│   ├── customer_transaction_summaries.json
│   ├── insights_summary.json
│   ├── raw_transactions_by_customer.json
│   ├── top_merchants_by_category.json
│   └── top_merchants_by_month.json
├── static/                      # Static files
├── templates/                   # HTML templates
├── all_summaries.py            # Data summary generator
├── database.py                 # Database connection
├── main.py                     # FastAPI application entry point
├── sql_db_agent.py             # SQL database agent
├── test_chat.py                # Testing script
├── travel_api_agent.py         # Travel-specific agent
└── requirements.txt            # Python dependencies
```

## 🔧 Core Components

### 1. SQL Database Agent (`sql_db_agent.py`)
- Handles natural language to SQL conversion
- Provides personalized financial analysis
- Generates charts and visualizations
- Caches customer information for performance

### 2. Data Summary Generator (`all_summaries.py`)
- Processes raw transaction data
- Generates customer-specific summaries
- Creates merchant and category analysis
- Produces financial insights

### 3. API Services (`api/services/app_services.py`)
- Chat functionality with AI
- Budget analysis tools
- Spending analyzer
- Raw transaction retrieval

### 4. AI Agents (`api/utils/agent.py`)
- Transaction filtering agent
- Context-aware insight selection
- Natural language processing

## 🌐 API Endpoints

### Health Check
- `GET /health` - Application health status

### Chat & Analysis
- `POST /chat` - Main chat endpoint with AI analysis
- `POST /chat/old` - Legacy chat endpoint
- `GET /spends-analyzer` - Spending analysis tool
- `GET /budget-tool` - Budget analysis tool
- `GET /raw-transactions` - Raw transaction data

### Data Retrieval
- `GET /customer-summary/{customer_id}` - Customer summary
- `GET /merchant-summary/{customer_id}` - Merchant analysis
- `GET /insights/{customer_id}` - Financial insights

## 💬 Usage Examples

### Chat Interface
```python
# Example chat request
{
    "customer_id": "10002",
    "message": "What is my monthly spending on groceries?",
    "session_id": "unique_session_id"
}
```

### Direct API Calls
```python
import requests

# Get spending analysis
response = requests.get(
    "http://localhost:8000/spends-analyzer",
    params={"customer_id": "10002"}
)

# Get budget analysis
response = requests.get(
    "http://localhost:8000/budget-tool",
    params={"customer_id": "10002"}
)
```

### SQL Database Agent
```python
from sql_db_agent import DatabaseChain

# Initialize the agent
db_chain = DatabaseChain()

# Query with natural language
result = db_chain.query(
    question="What is my month wise spends in groceries?",
    customer_id="10002"
)

# Generate chart
chart_result = db_chain.generate_chart(
    question="Show my spending trends",
    customer_id="10002"
)
```

## 🔍 Data Analysis Features

### Transaction Analysis
- Monthly spending trends
- Category-wise breakdown
- Merchant analysis
- Income vs expense tracking
- Savings rate calculation

### AI-Powered Insights
- Spending pattern detection
- Anomaly identification
- Budget overrun alerts
- Savings opportunities
- Trend analysis

### Visualization
- Line charts for trends
- Bar charts for comparisons
- Interactive data tables
- Exportable reports

## 🛡️ Security Considerations

- API keys are stored in environment variables
- Database credentials are secured
- CORS is configured for web access
- Input validation on all endpoints
- Error handling prevents data leakage

## 🧪 Testing

### Run the test script
```bash
python test_chat.py
```

### Test individual components
```python
# Test database connection
from database import db
result = db.run("SELECT COUNT(*) FROM raw_transactions")

# Test AI agent
from sql_db_agent import DatabaseChain
agent = DatabaseChain()
result = agent.query("Show my total income", "10002")
```

## 📊 Data Format

### Transaction Data Schema
```json
{
    "transaction_id": "unique_id",
    "customer_id": "customer_identifier",
    "merchant_name": "Merchant Name",
    "transaction_date": "dd-mm-yyyy",
    "transaction_month": "yyyy-mm",
    "transaction_amount": 100.50,
    "transaction_type": "Income|Expense",
    "merchant_category": "Category Name",
    "is_online": true,
    "merchant_city": "City Name",
    "merchant_logo": "logo_url",
    "transaction_currency": "AED"
}
```

### API Response Format
```json
{
    "answer": "AI-generated response",
    "data": [
        {
            "column1": "value1",
            "column2": "value2"
        }
    ],
    "chart": {
        "xAxis": {
            "type": "category",
            "data": ["value1", "value2"]
        },
        "yAxis": {
            "type": "value"
        },
        "series": [
            {
                "name": "Series Name",
                "data": [10, 20],
                "type": "line"
            }
        ]
    }
}
```

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `MODEL` | OpenAI model to use | No | `gpt-3.5-turbo` |
| `DB_USER` | Database username | Yes | - |
| `DB_PASSWORD` | Database password | Yes | - |
| `DB_HOST` | Database host | Yes | - |
| `DB_PORT` | Database port | No | `5432` |
| `DB_NAME` | Database name | Yes | - |
| `DEBUG` | Debug mode | No | `False` |
| `LOG_LEVEL` | Logging level | No | `INFO` |

### Database Configuration
The application uses PostgreSQL with the following requirements:
- Connection pooling support
- JSON data type support
- Full-text search capabilities
- Transaction support

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations
- Use environment variables for all sensitive data
- Set up proper logging
- Configure database connection pooling
- Implement rate limiting
- Set up monitoring and alerting
- Use HTTPS in production
- Implement proper backup strategies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the documentation
- Review the API endpoints
- Test with the provided examples
- Check the logs for error messages

## 🔄 Updates and Maintenance

### Regular Maintenance Tasks
- Update dependencies regularly
- Monitor API usage and costs
- Backup database regularly
- Review and update security configurations
- Monitor application performance

### Version Updates
- Check for breaking changes in dependencies
- Test thoroughly before deployment
- Update documentation as needed
- Notify users of significant changes

---

**Note**: This application requires a valid OpenAI API key and properly configured PostgreSQL database to function. Make sure to set up all environment variables before running the application. 