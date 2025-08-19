import os
from crewai import Crew
from crew.agents.ingestion_agent import create_ingestion_agent
from crew.tasks.ingestion_task import create_ingestion_task


def process_csv_file(file_path: str, required_columns: list = None):
    """
    Simple function to process a CSV file using the Ingestion Agent
    
    Args:
        file_path: Path to the CSV file
        required_columns: List of required column names (optional)
    
    Returns:
        Processing result
    """
    
    # Create agent and task
    agent = create_ingestion_agent()
    task = create_ingestion_task(agent, file_path, required_columns)
    
    # Create and run crew
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    # Execute the task
    result = crew.kickoff()
    return result


def main():
    """Main function - example usage"""
    
    # Set up OpenAI API key (required for CrewAI)
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("   You can set it in a .env file or as an environment variable")
        return
    
    print("=== Ingestion Agent Demo ===\n")
    
    # Example 1: Process a simple CSV
    try:
        result = process_csv_file(
            file_path="data/sample_data/clean_sales.csv",
            required_columns=["Date", "Product", "Sales Amount"]
        )
        
        print("\n✅ Processing completed!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")


if __name__ == "__main__":
    main()