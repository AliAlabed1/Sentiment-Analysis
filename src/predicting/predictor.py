import os
import sys
import pandas as pd
import joblib
# Add the directory containing the `src` folder to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))  # Adjust to reach `src`
sys.path.append(PROJECT_ROOT)
current_path = os.getcwd()
worksapce = current_path.split('src')[0]
from utils.logging_utils.app_logger import app_logger
from data_pipline.pipeline import data_pipeline


class Predict:
    '''
        This class is for make a prediction
    '''
    def make_prediction(self,text,model):
        '''
            This function is to make prediction
            params:
                text:str
                model:sklearn model
            return:
                label: 0,1
        '''
        app_logger.info('Starting prediction...')
        app_logger.info('Applying data pipe line...')
        predections_path = f'{worksapce}/Data/Predictions/news.csv'
        if os.path.exists(predections_path):
            df = pd.read_csv(predections_path)
        else:
            df = pd.DataFrame(columns = ['news','label'])
        row = pd.DataFrame([{
            "news": text,
        }])
        row.to_csv('temp.csv')
        pipeline = data_pipeline()
        df,vector = pipeline.apply('temp.csv', 'predict')
        app_logger.info('Finish data pipeline!')
        print(df)
        prediction = model.predict(vector)
        print(prediction)

if __name__ == '__main__':
    model = joblib.load(f'{worksapce}/Models/logistic_model.pkl')
    predictor = Predict()
    predictor.make_prediction('This is bad  news for who are wating the new year opening, the markeet is not good but you must wait for 24 hours',model)
        