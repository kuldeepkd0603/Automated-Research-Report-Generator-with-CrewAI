import pandas as pd
import chardet
import os
from typing import Optional, Tuple, Dict, Any
# from crewai_tools import BaseTool
# from crewai_tools import Tool
from crewai_tools.tools import BaseTool
from pydantic import BaseModel, Field


class CSVReaderInput(BaseModel):
    file_path: str = Field(..., description="Path to the CSV file")
    encoding: Optional[str] = Field(None, description="File encoding (auto-detect if None)")
    delimiter: Optional[str] = Field(None, description="CSV delimiter (auto-detect if None)")


class CSVReaderTool(BaseTool):
    name: str = "CSV Reader Tool"
    description: str = "Load CSV files with automatic encoding detection and format handling"
    args_schema: type[BaseModel] = CSVReaderInput
    
    def _run(self, file_path: str, encoding: Optional[str] = None, delimiter: Optional[str] = None) -> Dict[str, Any]:
        """
        Load CSV file with encoding detection and error handling
        
        Returns:
            Dict containing:
            - success: bool
            - data: pd.DataFrame or None
            - error: str or None
            - metadata: dict with file info
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "data": None,
                    "error": f"File not found: {file_path}",
                    "metadata": {}
                }
            
            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 100:  # 100MB limit
                return {
                    "success": False,
                    "data": None,
                    "error": f"File too large: {file_size_mb:.2f}MB (max 100MB)",
                    "metadata": {"file_size_mb": file_size_mb}
                }
            
            # Detect encoding if not provided
            if encoding is None:
                encoding = self._detect_encoding(file_path)
            
            # Detect delimiter if not provided
            if delimiter is None:
                delimiter = self._detect_delimiter(file_path, encoding)
            
            # Load CSV
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                delimiter=delimiter,
                low_memory=False
            )
            
            metadata = {
                "file_path": file_path,
                "file_size_mb": file_size_mb,
                "encoding": encoding,
                "delimiter": delimiter,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist()
            }
            
            return {
                "success": True,
                "data": df,
                "error": None,
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"Failed to read CSV: {str(e)}",
                "metadata": {}
            }
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet"""
        try:
            with open(file_path, 'rb') as f:
                # Read first 10KB for detection
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'
    
    def _detect_delimiter(self, file_path: str, encoding: str) -> str:
        """Detect CSV delimiter by analyzing first few lines"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                # Read first 1KB
                sample = f.read(1024)
                
            # Check common delimiters
            delimiters = [',', ';', '\t', '|']
            delimiter_counts = {}
            
            for delimiter in delimiters:
                count = sample.count(delimiter)
                if count > 0:
                    delimiter_counts[delimiter] = count
            
            # Return most frequent delimiter, default to comma
            if delimiter_counts:
                return max(delimiter_counts, key=delimiter_counts.get)
            return ','
            
        except Exception:
            return ','

