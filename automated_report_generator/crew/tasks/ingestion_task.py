from crewai import Task
from typing import Dict, Any
import pandas as pd


def create_ingestion_task(agent, file_path: str, required_columns: list = None) -> Task:
    """Create the ingestion task for processing CSV files"""
    
    return Task(
        description=f"""
        Process the CSV file located at: {file_path}
        
        Your task is to:
        1. Load the CSV file using the CSV Reader Tool with automatic encoding detection
        2. Validate the data structure and quality using the Validation Tool
        3. Clean the data using the Pandas Tool with these operations:
           - remove_duplicates
           - handle_missing_values  
           - normalize_column_names
           - infer_data_types
           - remove_empty_rows_cols
        4. Perform final validation to ensure data quality
        5. Return a clean DataFrame ready for analysis
        
        Required columns (if any): {required_columns or 'None specified'}
        
        If you encounter any critical issues:
        - Invalid file format: Return error with specific issue
        - Encoding problems: Try alternative encodings
        - Missing critical columns: Suggest similar columns if available
        - Too many missing values: Flag columns for review
        
        Always provide a detailed summary of what was processed and any issues found.
        """,
        
        expected_output="""
        A comprehensive report containing:
        1. Processing status (success/failure)
        2. Clean pandas DataFrame (if successful)
        3. Data quality metrics and validation results
        4. List of cleaning operations performed
        5. Any warnings or suggestions for improvement
        6. Metadata about the processed file
        
        Format the output as a structured dictionary with clear sections.
        """,
        
        agent=agent
    )