import os
import pandas as pd
import numpy as np
from crewai import Crew
from crew.agents.ingestion_agent import create_ingestion_agent
from crew.tasks.ingestion_task import create_ingestion_task

# ✅ Add Groq client
from groq import Groq

# Initialize Groq client (set your API key in environment variable GROQ_API_KEY)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def create_sample_data():
    """Create sample CSV files for testing"""
    os.makedirs("data/sample_data", exist_ok=True)
    
    # Sample clean dataset
    sales_data = {
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'Product': ['Widget A', 'Widget B', 'Widget A', 'Widget C', 'Widget B'],
        'Sales Amount': [100.50, 250.75, 150.25, 300.00, 200.50],
        'Region': ['North', 'South', 'East', 'West', 'North'],
        'Salesperson': ['John', 'Jane', 'Bob', 'Alice', 'John']
    }
    pd.DataFrame(sales_data).to_csv("data/sample_data/clean_sales.csv", index=False)

    print("✅ Sample data created in data/sample_data/")


def test_with_groq():
    """Test complete ingestion workflow with Groq"""
    print("\n=== Testing Complete Agent with Groq ===\n")

    try:
        # Quick check: call Groq with a simple prompt
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # pick available Groq model
            messages=[{"role": "user", "content": "Hello Groq, are you running?"}]
        )
        print("✅ Groq API is working")
        print("Groq response:", chat_completion.choices[0].message["content"])
    except Exception as e:
        print(f"❌ Could not connect to Groq: {str(e)}")
        print("   Make sure GROQ_API_KEY is set in your environment")
        return

    # Create sample data
    create_sample_data()

    # Create agent
    try:
        ingestion_agent = create_ingestion_agent()
        print("✅ Agent created successfully")
    except Exception as e:
        print(f"❌ Failed to create agent: {str(e)}")
        return

    # Test with clean data
    print("\n--- Testing with Clean Data ---")
    try:
        task = create_ingestion_task(
            agent=ingestion_agent,
            file_path="data/sample_data/clean_sales.csv",
            required_columns=["Date", "Product", "Sales Amount"]
        )

        crew = Crew(
            agents=[ingestion_agent],
            tasks=[task],
            verbose=True
        )

        result = crew.kickoff()
        print("\n✅ Clean data test completed!")
        print(f"Result summary: {str(result)[:200]}...")

    except Exception as e:
        print(f"\n❌ Clean data test failed: {str(e)}")


def run_all_tests():
    """Run tests (Groq version)"""
    print("=== Ingestion Agent Tests with Groq ===\n")

    test_with_groq()
    print("\nAll tests completed!")


if __name__ == "__main__":
    run_all_tests()
