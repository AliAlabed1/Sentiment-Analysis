import os 
import sys 
import pandas as pd
import joblib
import mlflow
import subprocess
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBClassifier
# Add the directory containing the `src` folder to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))  # Adjust to reach `src`
sys.path.append(PROJECT_ROOT)
current_path = os.getcwd()
worksapce = current_path.split('src')[0]

from training.trainer import train
from utils.evaluation_utils.evaluations import Evaluate
from utils.logging_utils.app_logger import app_logger

class MlFlow_Runner:
    def run(self):
        '''
        This function runs training process and run experments on MLFLOW
        '''
        app_logger.info('Starting MLFLow Runner ...')
        mlflow_process = subprocess.Popen(["mlflow", "ui", "--host", "127.0.0.1", "--port", "5000"], 
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(10)

        trainer = train()
        models = ['VADER','Logistic','Random','XGB']
        df = pd.read_csv(f'{worksapce}/Data/Cleaned dataset/cleaned_news.csv')
        df = df.dropna()
        X_train = joblib.load(f'{worksapce}/Data/Vectorized/X_train_tfidf.pkl')
        y_train = joblib.load(f'{worksapce}/Data/Vectorized/y_train.pkl')
        X_test = joblib.load(f'{worksapce}/Data/Vectorized/X_test_tfidf.pkl')
        y_test = joblib.load(f'{worksapce}/Data/Vectorized/y_test.pkl')
        evaluator = Evaluate()
        mlflow.set_experiment("Sentiment_Analysis")
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        for model in models:
            predictions, model_path = trainer.train_model(model,df,X_train,y_train,X_test)
            if model == 'VADER':
                accuracy,precesion,recall,f1 = evaluator.compute_metrics(predictions,df['label'],model)
            else:
                accuracy,precesion,recall,f1 = evaluator.compute_metrics(predictions,y_test,model)
            with mlflow.start_run(run_name=model):
                if model== 'XGB':
                    _model = XGBClassifier()
                    _model.load_model(model_path)
                    mlflow.xgboost.log_model(_model,model)
                elif model == 'VADER':
                    mlflow.sklearn.log_model(SentimentIntensityAnalyzer(),model)
                else:
                    _model = joblib.load(model_path)
                    mlflow.sklearn.log_model(_model,model)
                mlflow.log_metrics({
                    'accuracy':accuracy,
                    'recall':recall,
                    'precesion':precesion,
                    'f1-score':f1
                })
        app_logger.info("Training complete. MLflow UI is running at http://127.0.0.1:5000")

        # Optional: Keep script running so MLflow UI stays alive
        mlflow_process.wait()
        return mlflow_process

if __name__ == '__main__':
    runner = MlFlow_Runner()
    runner.run()