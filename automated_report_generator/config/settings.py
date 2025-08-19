import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Data validation settings
    MAX_FILE_SIZE_MB = 100
    SUPPORTED_ENCODINGS = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    # Missing value thresholds
    MAX_MISSING_PERCENTAGE = 0.3  # 30% missing values max per column
    
    # Required columns (can be configured per use case)
    CRITICAL_COLUMNS = []  # Will be set based on data context
    
    # Data type inference settings
    NUMERIC_THRESHOLD = 0.8  # 80% of values must be numeric to infer numeric type
    
    # OpenAI settings (for future LLM integration)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")