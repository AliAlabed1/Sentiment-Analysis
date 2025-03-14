import os
import sys
import joblib
import pandas as pd
# Add the directory containing the `src` folder to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))  # Adjust to reach `src`
sys.path.append(PROJECT_ROOT)
current_path = os.getcwd()
worksapce = current_path.split('src')[0]

from utils.evaluation_utils.evaluations import Evaluate
from utils.logging_utils.app_logger import app_logger
from training.trainer import train
from data_pipline.pipeline import data_pipeline


if __name__ == '__main__':
    try:

        models = ['VADER','Logistic','Random','XGB']
        pipeline = data_pipeline()
        # pipeline.apply(f'{worksapce}/Data/Source_dataset/news.csv')
        df = pd.read_csv(f'{worksapce}/Data/Cleaned dataset/cleaned_news.csv')
        df = df.dropna()
        trainer = train()
        X_train = joblib.load(f'{worksapce}/Data/Vectorized/X_train_tfidf.pkl')
        X_test = joblib.load(f'{worksapce}/Data/Vectorized/X_test_tfidf.pkl')
        y_train = joblib.load(f'{worksapce}/Data/Vectorized/y_train.pkl')
        y_test = joblib.load(f'{worksapce}/Data/Vectorized/y_test.pkl')
        trainer = train()
        evaluator = Evaluate()
        results_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
        for model in models:
            predictions = trainer.train_model(model,df,X_train,y_train,X_test)
            if model == 'VADER':
                accuracy,precesion,recall,f1 = evaluator.compute_metrics(predictions,df['label'],model)
            else:
                accuracy,precesion,recall,f1 = evaluator.compute_metrics(predictions,y_test,model)
            new_row = pd.DataFrame([{
                "Model": model,
                "Accuracy": accuracy,
                "Precision": precesion,
                "Recall": recall,
                "F1-Score": f1
            }])

            # Append the new row using `pd.concat()`
            results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df.to_csv(f"{worksapce}/Data/evaluation_results.csv", index=False)
            

    except Exception as e:
        app_logger.error(f"Exception: {e}")



