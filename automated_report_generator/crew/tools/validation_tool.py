import pandas as pd
from typing import Dict, Any, List, Optional
from crewai.tools import BaseTool  # Updated import
from pydantic import BaseModel, Field
from config.settings import Settings


class ValidationToolInput(BaseModel):
    required_columns: Optional[List[str]] = Field(None, description="List of required column names")
    validation_rules: Optional[Dict[str, Any]] = Field(None, description="Custom validation rules")


class ValidationTool(BaseTool):
    name: str = "Data Validation Tool"
    description: str = "Validate DataFrame schema and data quality"
    args_schema: type[BaseModel] = ValidationToolInput
    
    def __init__(self):
        super().__init__()
        self.current_dataframe = None
    
    def set_dataframe(self, df: pd.DataFrame):
        """Set the current dataframe to work with"""
        self.current_dataframe = df
    
    def _run(self, required_columns: Optional[List[str]] = None, 
             validation_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate DataFrame and return detailed validation report
        """
        try:
            if self.current_dataframe is None:
                return {
                    "is_valid": False,
                    "errors": ["No dataframe set. Use set_dataframe() first."],
                    "warnings": [],
                    "suggestions": [],
                    "data_quality_score": 0.0,
                    "summary": {}
                }
            
            validation_report = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "suggestions": [],
                "data_quality_score": 0.0,
                "summary": {}
            }
            
            # Basic validations
            self._validate_basic_structure(self.current_dataframe, validation_report)
            self._validate_required_columns(self.current_dataframe, required_columns or [], validation_report)
            self._validate_data_quality(self.current_dataframe, validation_report)
            self._validate_data_types(self.current_dataframe, validation_report)
            
            # Custom validations if provided
            if validation_rules:
                self._apply_custom_validations(self.current_dataframe, validation_rules, validation_report)
            
            # Calculate overall data quality score
            validation_report["data_quality_score"] = self._calculate_quality_score(self.current_dataframe, validation_report)
            
            # Set overall validity
            validation_report["is_valid"] = len(validation_report["errors"]) == 0
            
            return validation_report
            
        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "suggestions": [],
                "data_quality_score": 0.0,
                "summary": {}
            }
    
    def _validate_basic_structure(self, df: pd.DataFrame, report: Dict[str, Any]):
        """Validate basic DataFrame structure"""
        if df.empty:
            report["errors"].append("DataFrame is empty")
            return
        
        if len(df.columns) == 0:
            report["errors"].append("DataFrame has no columns")
            return
        
        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            report["errors"].append(f"Duplicate column names found: {duplicate_cols}")
        
        # Update summary
        report["summary"]["total_rows"] = len(df)
        report["summary"]["total_columns"] = len(df.columns)
        report["summary"]["memory_usage_mb"] = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    def _validate_required_columns(self, df: pd.DataFrame, required_columns: List[str], report: Dict[str, Any]):
        """Validate that required columns are present"""
        if not required_columns:
            return
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            report["errors"].append(f"Missing required columns: {missing_columns}")
            
            # Suggest similar column names
            suggestions = self._suggest_similar_columns(missing_columns, df.columns.tolist())
            if suggestions:
                report["suggestions"].extend(suggestions)
    
    def _validate_data_quality(self, df: pd.DataFrame, report: Dict[str, Any]):
        """Validate data quality metrics"""
        quality_issues = []
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = missing_count / len(df)
            
            if missing_percentage > Settings.MAX_MISSING_PERCENTAGE:
                quality_issues.append(f"Column '{column}' has {missing_percentage:.1%} missing values")
            elif missing_percentage > 0.1:  # 10% threshold for warnings
                report["warnings"].append(f"Column '{column}' has {missing_percentage:.1%} missing values")
        
        if quality_issues:
            report["errors"].extend(quality_issues)
        
        # Check for completely empty columns
        empty_columns = [col for col in df.columns if df[col].isnull().all()]
        if empty_columns:
            report["warnings"].append(f"Completely empty columns: {empty_columns}")
            report["suggestions"].append("Consider removing empty columns")
        
        # Update summary with quality metrics
        report["summary"]["missing_values_per_column"] = df.isnull().sum().to_dict()
        report["summary"]["duplicate_rows"] = df.duplicated().sum()
    
    def _validate_data_types(self, df: pd.DataFrame, report: Dict[str, Any]):
        """Validate and suggest data type improvements"""
        type_suggestions = []
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check if numeric conversion is possible
                try:
                    numeric_converted = pd.to_numeric(df[column], errors='coerce')
                    non_null_original = df[column].notna().sum()
                    non_null_converted = numeric_converted.notna().sum()
                    
                    if non_null_converted / non_null_original > 0.8:  # 80% convertible
                        type_suggestions.append(f"Column '{column}' could be converted to numeric")
                except:
                    pass
                
                # Check if datetime conversion is possible
                if self._could_be_datetime(df[column]):
                    type_suggestions.append(f"Column '{column}' might be a date/time field")
        
        if type_suggestions:
            report["suggestions"].extend(type_suggestions)
        
        # Update summary with data types
        report["summary"]["data_types"] = df.dtypes.astype(str).to_dict()
    
    def _apply_custom_validations(self, df: pd.DataFrame, rules: Dict[str, Any], report: Dict[str, Any]):
        """Apply custom validation rules"""
        for rule_name, rule_config in rules.items():
            try:
                if rule_name == "value_range":
                    self._validate_value_range(df, rule_config, report)
                elif rule_name == "unique_values":
                    self._validate_unique_values(df, rule_config, report)
                elif rule_name == "pattern_match":
                    self._validate_pattern_match(df, rule_config, report)
            except Exception as e:
                report["warnings"].append(f"Custom validation '{rule_name}' failed: {str(e)}")
    
    def _validate_value_range(self, df: pd.DataFrame, config: Dict[str, Any], report: Dict[str, Any]):
        """Validate numeric values are within expected ranges"""
        for column, range_config in config.items():
            if column not in df.columns:
                continue
            
            min_val = range_config.get('min')
            max_val = range_config.get('max')
            
            if min_val is not None:
                violations = (df[column] < min_val).sum()
                if violations > 0:
                    report["errors"].append(f"Column '{column}': {violations} values below minimum {min_val}")
            
            if max_val is not None:
                violations = (df[column] > max_val).sum()
                if violations > 0:
                    report["errors"].append(f"Column '{column}': {violations} values above maximum {max_val}")
    
    def _validate_unique_values(self, df: pd.DataFrame, config: Dict[str, List[str]], report: Dict[str, Any]):
        """Validate that specified columns have unique values"""
        for column in config.get('columns', []):
            if column not in df.columns:
                continue
            
            duplicates = df[column].duplicated().sum()
            if duplicates > 0:
                report["errors"].append(f"Column '{column}' should be unique but has {duplicates} duplicates")
    
    def _validate_pattern_match(self, df: pd.DataFrame, config: Dict[str, Any], report: Dict[str, Any]):
        """Validate that values match expected patterns (regex)"""
        import re
        
        for column, pattern in config.items():
            if column not in df.columns:
                continue
            
            try:
                regex = re.compile(pattern)
                non_matches = ~df[column].astype(str).str.match(regex)
                violations = non_matches.sum()
                
                if violations > 0:
                    report["warnings"].append(f"Column '{column}': {violations} values don't match pattern {pattern}")
            except Exception as e:
                report["warnings"].append(f"Pattern validation for '{column}' failed: {str(e)}")
    
    def _calculate_quality_score(self, df: pd.DataFrame, report: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0
        
        # Deduct for errors
        score -= len(report["errors"]) * 10
        
        # Deduct for warnings
        score -= len(report["warnings"]) * 5
        
        # Deduct for missing values
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        if total_cells > 0:
            missing_percentage = missing_cells / total_cells
            score -= missing_percentage * 30
        
        # Deduct for duplicates
        duplicate_percentage = df.duplicated().sum() / len(df) if len(df) > 0 else 0
        score -= duplicate_percentage * 20
        
        return max(0.0, score)
    
    def _suggest_similar_columns(self, missing_columns: List[str], available_columns: List[str]) -> List[str]:
        """Suggest similar column names using simple string similarity"""
        suggestions = []
        
        for missing_col in missing_columns:
            best_matches = []
            missing_lower = missing_col.lower()
            
            for available_col in available_columns:
                available_lower = available_col.lower()
                
                # Simple similarity checks
                if missing_lower in available_lower or available_lower in missing_lower:
                    best_matches.append(available_col)
                elif self._levenshtein_distance(missing_lower, available_lower) <= 2:
                    best_matches.append(available_col)
            
            if best_matches:
                suggestions.append(f"For missing column '{missing_col}', consider: {best_matches}")
        
        return suggestions
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _could_be_datetime(self, series: pd.Series) -> bool:
        """Check if a series could be datetime (same logic as in PandasTool)"""
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        
        date_indicators = ['/', '-', 'T', ':', ' ']
        for value in sample:
            str_value = str(value)
            if any(indicator in str_value for indicator in date_indicators):
                if len(str_value) > 6:
                    return True
        return False