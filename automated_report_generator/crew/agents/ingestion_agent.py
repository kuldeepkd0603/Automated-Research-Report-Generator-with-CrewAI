from crewai import Agent, LLM
from tools.csv_reader_tool import CSVReaderTool
from tools.pandas_tool import PandasTool
from tools.validation_tool import ValidationTool
from config.settings import Settings


def create_ingestion_agent() -> Agent:
    """Create the Ingestion Agent with required tools"""
    
    # Configure LLM to use Ollama
    llm = LLM(
        model=f"ollama/{Settings.MODEL_NAME}",
        base_url=Settings.OLLAMA_BASE_URL
    )
    
    return Agent(
        role="Data Gateway & Validator",
        goal="Transform raw CSV files into clean, analysis-ready DataFrames",
        backstory="""You are an expert data engineer specializing in data ingestion and validation. 
        Your job is to take messy, real-world CSV files and transform them into clean, reliable datasets 
        that analysts can trust. You have extensive experience with data quality issues, encoding problems, 
        and inconsistent formats. You always validate data thoroughly and provide clear feedback about 
        any issues found.""",
        
        tools=[
            CSVReaderTool(),
            PandasTool(),
            ValidationTool()
        ],
        
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )
