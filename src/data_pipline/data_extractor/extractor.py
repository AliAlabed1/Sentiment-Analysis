import os
import pandas as pd
import sys

# Add the directory containing the `src` folder to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))  # Adjust to reach `src`
sys.path.append(PROJECT_ROOT)

from utils.logging_utils.app_logger import app_logger


current_path = os.getcwd()
worksapce = current_path.split('src')[0]

class LoadCSVFile():
    def load_data(self,path:str):
        """
        Loads data  from the specified path, handling errors 
        and logging any issues encountered during the loading process.

        Parameters:
        - path (str): The loaded CSV data file.

        Raises:
        - FileNotFoundError: If the specified data path does not exist.
        - ValueError: If the loaded df is None.
        - OSError: If there is an issue loading the CSV file.  
        """

        try:
            if not os.path.exists(path):
                app_logger.error(f"The file does not exist: {path}")
                raise FileNotFoundError(f"The specified CSV file path does not exist: {path}")
            
            df = pd.read_csv(path)

            if df is None:
                raise ValueError("The loaded df is None. Please check the CSV file.")
            app_logger.info(f"df loaded successfully from {path}")
            return df
        except Exception as e:
            # Catch any other exceptions that may occur
            app_logger.error(f"An unexpected error occurred while loading the CSV file: {str(e)}")
            raise


# Test the Class and main Funcation 
if __name__ == "__main__":
    iload = LoadCSVFile()
    df = iload.load_data(f"{worksapce}/Data/results.csv")