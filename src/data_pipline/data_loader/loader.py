import os 
import sys

# Add the directory containing the `src` folder to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))  # Adjust to reach `src`
sys.path.append(PROJECT_ROOT)
current_path = os.getcwd()
worksapce = current_path.split('src')[0]
from utils.logging_utils.app_logger import app_logger

class Load:
    '''
        This class is to save cleaned data and vectorize the cleaned text
    '''

    def save_data(self,df):
        df.to_csv(f'{worksapce}/Data/Cleaned dataset/cleaned_news.csv')
        app_logger.info("Save cleaned data is done successfully!")