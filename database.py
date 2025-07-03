import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
load_dotenv('.env')

class DatabaseConnection:

    def __init__(self) -> SQLDatabase:
        """Initialize PostgreSQL database connection using SQLAlchemy URI."""
        try:
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT", "5432")
            db_name = os.getenv("DB_NAME")

            # SQLAlchemy URI format for PostgreSQL
            pg_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            
            self.db = SQLDatabase.from_uri(pg_uri)
        except Exception as e:
            raise Exception(f"Failed to initialize database: {str(e)}")

db = DatabaseConnection().db
