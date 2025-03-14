import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add the directory containing the `src` folder to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))  # Adjust to reach `src`
sys.path.append(PROJECT_ROOT)
current_path = os.getcwd()
worksapce = current_path.split('src')[0]

from utils.logging_utils.app_logger import app_logger

class Evaluate:
    '''
        This class is to Evaluate models.
    '''
    def compute_metrics(self,predictions,y_test,model_name):
        '''
            This function is to compute accuracy,f1-score,recall,precession
            params:
                predictions: list of [0,1]
                y_test: list of [0,1]
                model_name:str
            return:
                accuracy,presecion,recal,f1-score: numbers
        '''
        app_logger.info(f'Start Evaluating model {model_name}')
        accuracy = accuracy_score(y_test, predictions)
        precesion = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        app_logger.info(f'Evaluating Reults for model {model_name}\nprecesion:{precesion}\nRecall:{recall}\naccuracy:{accuracy}\nf1-score:{f1}')
        return accuracy,precesion,recall,f1

        




