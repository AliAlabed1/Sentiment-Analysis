import os
import sys

# Add the directory containing the `src` folder to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))  # Adjust to reach `src`
sys.path.append(PROJECT_ROOT)

from utils.logging_utils.app_logger import app_logger
from data_pipline.data_extractor.extractor import LoadCSVFile
from data_pipline.data_transformer.transformer import transform
from data_pipline.data_loader.loader import Load
class data_pipeline:

    def apply(self,path):
        try:
            if not isinstance(path,str):
                raise TypeError('Path is not a string type')
            extractor = LoadCSVFile()
            df = extractor.load_data(path)
            transformer = transform()
            df = transformer.clean_data(df)
            iloader = Load()
            iloader.save_data(df)

        except Exception as e:
            app_logger.error(f'Exeption:{e}')

if __name__ == '__main__':
    
    pipline = data_pipeline()
    pipline.apply('C:/Users/USER/Desktop/Qufzah/project/Data/Source_dataset/news.csv')

