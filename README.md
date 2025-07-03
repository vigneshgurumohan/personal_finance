# Finance Assistant Application

This application provides a finance assistant that analyzes customer transaction data and provides insights through a chat interface.

## Setup

### 1. Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Add other environment variables as needed
# DATABASE_URL=your_database_url_here
# SECRET_KEY=your_secret_key_here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Data Preparation

Ensure you have the required data files:
- `data/raw_transactions.csv` - Transaction data
- `summaries/` directory - Generated summary files

### 4. Generate Summaries

Run the summary generation script:

```bash
python all_summaries.py
```

### 5. Start the Application

```bash
python main.py
```

## Security Notes

- The `.env` file is already included in `.gitignore` to prevent accidentally committing sensitive information
- Never commit API keys or other sensitive credentials to version control
- Always use environment variables for configuration in production

## API Key Security

The application now properly reads the OpenAI API key from environment variables instead of hardcoding it. This ensures:

1. **Security**: API keys are not exposed in the source code
2. **Flexibility**: Different environments can use different API keys
3. **Best Practices**: Follows security best practices for credential management

## Files Modified

The following files were updated to use environment variables:

- `test_chat.py` - Updated to read API key from environment
- `all_summaries.py` - Updated to read API key from environment  
- `main.py` - Updated error message to reference `.env` file
- `.gitignore` - Already includes `.env` to prevent accidental commits 