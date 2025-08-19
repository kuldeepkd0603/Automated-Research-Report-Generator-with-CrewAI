import os
import pandas as pd
import numpy as np
from crewai import Crew
from tests.crew.agents.ingestion_agent import create_ingestion_agent
from tests.crew.tasks.ingestion_task import create_ingestion_task


def create_sample_data():
    """Create sample CSV files for testing"""
    
    # Create data directory if it doesn't exist
    os.makedirs("data/sample_data", exist_ok=True)
    
    # Sample 1: Clean sales data
    sales_data = {
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'Product': ['Widget A', 'Widget B', 'Widget A', 'Widget C', 'Widget B'],
        'Sales Amount': [100.50, 250.75, 150.25, 300.00, 200.50],
        'Region': ['North', 'South', 'East', 'West', 'North'],
        'Salesperson': ['John', 'Jane', 'Bob', 'Alice', 'John']
    }
    pd.DataFrame(sales_data).to_csv("data/sample_data/clean_sales.csv", index=False)
    
    # Sample 2: Messy data with issues
    messy_data = {
        'Date ': ['2024/01/01', '2024/01/02', '', '2024-01-04', 'invalid'],  # Trailing space, mixed formats
        ' Product Name': ['Widget A', 'Widget B', 'Widget A', None, 'Widget B'],  # Leading space, nulls
        'Sales$Amount': ['100.50', 'invalid', '150.25', '300', '200.50'],  # Mixed types
        'Region': ['North', 'South', 'East', 'West', 'North'],
        'Notes': ['', '', '', '', ''],  
        'Duplicate Col': ['A', 'B', 'C', 'D', 'E'],
        'Duplicate Col': ['X', 'Y', 'Z', 'W', 'V']  # Duplicate column name
    }
    
    # Add some duplicate rows
    messy_df = pd.DataFrame(messy_data)
    messy_df = pd.concat([messy_df, messy_df.iloc[0:2]], ignore_index=True)  # Add duplicates
    messy_df.to_csv("data/sample_data/messy_sales.csv", index=False)
    
    # Sample 3: Large dataset with missing values
    np.random.seed(42)
    large_data = {
        'ID': range(1000),
        'Value1': np.random.normal(100, 20, 1000),
        'Value2': np.random.uniform(0, 1000, 1000),
        'Category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'Text_Field': ['Sample text ' + str(i) for i in range(1000)]
    }
    
    large_df = pd.DataFrame(large_data)
    
    # Introduce missing values randomly
    missing_indices = np.random.choice(1000, 150, replace=False)
    large_df.loc[missing_indices, 'Value1'] = np.nan
    
    missing_indices = np.random.choice(1000, 80, replace=False) 
    large_df.loc[missing_indices, 'Category'] = np.nan
    
    large_df.to_csv("data/sample_data/large_dataset.csv", index=False)
    
    print("Sample data files created:")
    print("- data/sample_data/clean_sales.csv (clean data)")
    print("- data/sample_data/messy_sales.csv (data with issues)")
    print("- data/sample_data/large_dataset.csv (large dataset with missing values)")


def test_ingestion_agent():
    """Test the Ingestion Agent with different CSV files"""
    
    print("=== Testing Ingestion Agent ===\n")
    
    # Create sample data
    create_sample_data()
    
    # Create agent
    ingestion_agent = create_ingestion_agent()
    
    # Test cases
    test_cases = [
        {
            "name": "Clean Sales Data",
            "file": "data/sample_data/clean_sales.csv",
            "required_columns": ["Date", "Product", "Sales Amount"]
        },
        {
            "name": "Messy Sales Data", 
            "file": "data/sample_data/messy_sales.csv",
            "required_columns": ["Date", "Product Name", "Sales Amount"]
        },
        {
            "name": "Large Dataset",
            "file": "data/sample_data/large_dataset.csv",
            "required_columns": ["ID", "Value1"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"TEST {i}: {test_case['name']}")
        print(f"{'='*50}")
        
        # Create task
        task = create_ingestion_task(
            agent=ingestion_agent,
            file_path=test_case["file"],
            required_columns=test_case["required_columns"]
        )
        
        # Create crew and execute
        crew = Crew(
            agents=[ingestion_agent],
            tasks=[task],
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            print(f"\n✅ Test {i} completed successfully!")
            print(f"Result: {result}")
            
        except Exception as e:
            print(f"\n❌ Test {i} failed with error: {str(e)}")
        
        print("\n" + "-"*50)


def test_individual_tools():
    """Test individual tools separately for debugging"""
    
    print("\n=== Testing Individual Tools ===\n")
    
    # Ensure sample data exists
    create_sample_data()
    
    # Test CSV Reader Tool
    print("1. Testing CSV Reader Tool...")
    from tools.csv_reader_tool import CSVReaderTool
    
    csv_tool = CSVReaderTool()
    result = csv_tool._run("data/sample_data/clean_sales.csv")
    
    if result["success"]:
        print("✅ CSV Reader Tool working correctly")
        print(f"   Loaded {result['metadata']['rows']} rows, {result['metadata']['columns']} columns")
        print(f"   Columns: {result['metadata']['column_names']}")
    else:
        print(f"❌ CSV Reader Tool failed: {result['error']}")
    
    # Test Pandas Tool
    print("\n2. Testing Pandas Tool...")
    from tools.pandas_tool import PandasTool
    
    if result["success"]:
        pandas_tool = PandasTool()
        cleaning_result = pandas_tool._run(
            result["data"], 
            ["remove_duplicates", "normalize_column_names", "infer_data_types"]
        )
        
        if cleaning_result["success"]:
            print("✅ Pandas Tool working correctly")
            print("   Applied operations:", cleaning_result["applied_operations"])
        else:
            print(f"❌ Pandas Tool failed: {cleaning_result['error']}")
    
    # Test Validation Tool
    print("\n3. Testing Validation Tool...")
    from tools.validation_tool import ValidationTool
    
    if result["success"]:
        validation_tool = ValidationTool()
        validation_result = validation_tool._run(
            result["data"],
            required_columns=["Date", "Product"]
        )
        
        print(f"✅ Validation Tool completed")
        print(f"   Valid: {validation_result['is_valid']}")
        print(f"   Quality Score: {validation_result['data_quality_score']:.1f}")
        if validation_result["errors"]:
            print(f"   Errors: {validation_result['errors']}")
        if validation_result["warnings"]:
            print(f"   Warnings: {validation_result['warnings']}")


def test_error_scenarios():
    """Test error handling scenarios"""
    
    print("\n=== Testing Error Scenarios ===\n")
    
    # Create problematic files
    os.makedirs("data/test_errors", exist_ok=True)
    
    # Empty file
    with open("data/test_errors/empty.csv", "w") as f:
        f.write("")
    
    # Invalid CSV structure
    with open("data/test_errors/invalid.csv", "w") as f:
        f.write("This is not a CSV file\nIt has no structure\nAnd invalid content")
    
    # Non-existent file test
    error_tests = [
        {"name": "Non-existent File", "file": "data/test_errors/doesnt_exist.csv"},
        {"name": "Empty File", "file": "data/test_errors/empty.csv"},
        {"name": "Invalid CSV", "file": "data/test_errors/invalid.csv"}
    ]
    
    from tools.csv_reader_tool import CSVReaderTool
    csv_tool = CSVReaderTool()
    
    for test in error_tests:
        print(f"Testing: {test['name']}")
        result = csv_tool._run(test["file"])
        
        if not result["success"]:
            print(f"✅ Correctly handled error: {result['error']}")
        else:
            print(f"⚠️  Expected error but got success: {test['name']}")
        print()


if __name__ == "__main__":
    # Set up environment
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"  # Replace with actual key
    
    print("Starting Ingestion Agent Tests...")
    
    # Test individual tools first
    test_individual_tools()
    
    # Test error scenarios
    test_error_scenarios()
    
    # Test full agent (comment out if no OpenAI API key)
    # test_ingestion_agent()
    
    print("\n🎉 All tests completed!")