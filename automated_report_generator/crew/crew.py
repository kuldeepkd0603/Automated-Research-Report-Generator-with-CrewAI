from crew.tasks.ingestion_task import IngestionTask

class AutomatedReportCrew:
    def __init__(self, input_file):
        self.ingestion_task = IngestionTask(input_file)

    def run(self):
        df = self.ingestion_task.run()
        return df