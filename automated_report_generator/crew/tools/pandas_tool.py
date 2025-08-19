import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from crewai.tools import BaseTool  # Updated import
from pydantic import BaseModel, Field


class PandasToolInput(BaseModel):
    operations: List[str] = Field(..., description="List of cleaning operations to perform")
    # Note: dataframe will be passed as a string representation for the LLM


class PandasTool(BaseTool):
    name: str = "Pandas Data Cleaning Tool"
    description: str = "Clean and transform pandas DataFrames with various operations"
    args_schema: type[BaseModel] = PandasToolInput
    
    def __init__(self):
        super().__init__()
        self.current_dataframe = None
    
    def set_dataframe(self, df: pd.DataFrame):
        """Set the current dataframe to work with"""
        self.current_dataframe = df
    
    def _run(self, operations: List[str]) -> Dict[str, Any]:
        """
        Perform data cleaning operations on DataFrame
        
        Available operations:
        - remove_duplicates
        - handle_missing_values
        - normalize_column_names
        - infer_data_types
        - remove_empty_rows_cols
        """
        try:
            if self.current_dataframe is None:
                return {
                    "success": False,
                    "data": None,
                    "applied_operations": [],
                    "error": "No dataframe set. Use set_dataframe() first."
                }
            
            df = self.current_dataframe.copy()
            applied_operations = []
            
            for operation in operations:
                if operation == "remove_duplicates":
                    initial_rows = len(df)
                    df = df.drop_duplicates()
                    removed = initial_rows - len(df)
                    applied_operations.append(f"Removed {removed} duplicate rows")
                
                elif operation == "handle_missing_values":
                    missing_info = self._handle_missing_values(df)
                    applied_operations.append(f"Handled missing values: {missing_info}")
                
                elif operation == "normalize_column_names":
                    df.columns = self._normalize_column_names(df.columns)
                    applied_operations.append("Normalized column names")
                
                elif operation == "infer_data_types":
                    type_changes = self._infer_data_types(df)
                    applied_operations.append(f"Inferred data types: {type_changes}")
                
                elif operation == "remove_empty_rows_cols":
                    initial_shape = df.shape
                    df = self._remove_empty_rows_cols(df)
                    final_shape = df.shape
                    applied_operations.append(f"Shape changed from {initial_shape} to {final_shape}")
            
            return {
                "success": True,
                "data": df,
                "applied_operations": applied_operations,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": self.current_dataframe,
                "applied_operations": [],
                "error": f"Cleaning failed: {str(e)}"
            }
    
    def _handle_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Handle missing values based on column types"""
        missing_info = {}
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = missing_count / len(df)
            
            if missing_count > 0:
                if missing_percentage > 0.5:  # More than 50% missing
                    # Consider dropping column
                    missing_info[column] = f"High missing rate ({missing_percentage:.1%}) - consider dropping"
                elif df[column].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    df[column].fillna(df[column].median(), inplace=True)
                    missing_info[column] = f"Filled {missing_count} missing values with median"
                else:
                    # Fill categorical columns with mode or 'Unknown'
                    if not df[column].mode().empty:
                        df[column].fillna(df[column].mode()[0], inplace=True)
                        missing_info[column] = f"Filled {missing_count} missing values with mode"
                    else:
                        df[column].fillna('Unknown', inplace=True)
                        missing_info[column] = f"Filled {missing_count} missing values with 'Unknown'"
        
        return missing_info
    
    def _normalize_column_names(self, columns: pd.Index) -> List[str]:
        """Normalize column names: lowercase, replace spaces with underscores, remove special chars"""
        normalized = []
        for col in columns:
            # Convert to string and lowercase
            clean_name = str(col).lower()
            # Replace spaces and special characters with underscores
            clean_name = ''.join(c if c.isalnum() else '_' for c in clean_name)
            # Remove multiple consecutive underscores
            clean_name = '_'.join(filter(None, clean_name.split('_')))
            normalized.append(clean_name)
        return normalized
    
    def _infer_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Infer and convert appropriate data types"""
        type_changes = {}
        
        for column in df.columns:
            original_type = str(df[column].dtype)
            
            # Try to convert to numeric if possible
            if df[column].dtype == 'object':
                # Try numeric conversion
                numeric_series = pd.to_numeric(df[column], errors='coerce')
                if numeric_series.notna().sum() / len(df[column]) > 0.8:  # 80% can be converted
                    df[column] = numeric_series
                    type_changes[column] = f"{original_type} -> {df[column].dtype}"
                
                # Try datetime conversion
                elif self._could_be_datetime(df[column]):
                    try:
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                        type_changes[column] = f"{original_type} -> datetime64[ns]"
                    except:
                        pass
        
        return type_changes
    
    def _could_be_datetime(self, series: pd.Series) -> bool:
        """Check if a series could be datetime"""
        # Simple heuristic: check if series contains date-like patterns
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        
        date_indicators = ['/', '-', 'T', ':', ' ']
        for value in sample:
            str_value = str(value)
            if any(indicator in str_value for indicator in date_indicators):
                if len(str_value) > 6:  # Minimum reasonable date length
                    return True
        return False
    
    def _remove_empty_rows_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove completely empty rows and columns"""
        # Remove empty rows
        df = df.dropna(how='all')
        # Remove empty columns
        df = df.dropna(axis=1, how='all')
        return df

