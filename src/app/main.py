import os
import sys
import joblib
import pandas as pd
import questionary
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
from MLflow.runner import MlFlow_Runner
from api.run_server import server
class Main:
    def choice_0(self):
        models = ['VADER','Logistic','Random','XGB']
        pipeline = data_pipeline()
        pipeline.apply(f'{worksapce}/Data/Source_dataset/news.csv','train')
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
    
    def choice_1(self):
        runner = MlFlow_Runner()
        runner.run()

    def choice_2(self):
        serve = server()
        serve.run()

    def run_choice(self,choice):
        mthod = getattr(self,f'choice_{choice}',None)
        if callable(mthod):
            mthod()
        else:
            raise Exception("Unexpected exeption accured!!")

if __name__ == '__main__':
    try:
        print('\n               ****************Sentiment Analysis********************\
              \n                        You can choose on of following choices:'
            )
        choices = ["Run the data pipline and train models from scratch","Run MLflow Runner","Run the app to predict"]
        choice = questionary.select(
            "Choose the process:",
            choices=choices,
            use_arrow_keys=True
        ).ask()
        main = Main()

        if choice == choices[0]:
            main.run_choice(0)
        elif choice == choices[1]:
            main.choice_1()
            
        elif choice == choices[2]:
            main.choice_2()
            
    except Exception as e:
        app_logger.error(f"Exception: {e}")



