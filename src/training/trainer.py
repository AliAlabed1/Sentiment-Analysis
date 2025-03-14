import os
import sys
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
 
# Add the directory containing the `src` folder to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))  # Adjust to reach `src`
sys.path.append(PROJECT_ROOT)
current_path = os.getcwd()
worksapce = current_path.split('src')[0]
from utils.logging_utils.app_logger import app_logger
from utils.evaluation_utils.evaluations import Evaluate

class train:
    analyzer = SentimentIntensityAnalyzer()

    def get_vader_sentiment(self,text):
        score = self.analyzer.polarity_scores(text)['compound']
        return 1 if score > 0 else 0
    

    def train_VADER(self,df):
        '''
            This function is to train VADER model
            params:
                df:dataframe
            return:
                predictions:array of {0,1}
        '''
        app_logger.info('Training VADER model....')
        predictions = df['cleaned_text'].apply(self.get_vader_sentiment)
        app_logger.info('Training Done!')
        return predictions
    
    def train_Logistic(self,X_train,y_train,X_test):
        '''
            This function is to train Logistic model
            params:
                X_train,X_test:array of vectorized text
                y_train: array of [0,1]
            return:
                predictions:array of {0,1}
        '''
        model_path = f"{worksapce}/Models/logistic_model.pkl"
        if os.path.exists(model_path):
            app_logger.info("Loading Existed Logistic Model...")
            model = joblib.load(model_path)
        else:
            app_logger.info("Initilise Logisitic Regression Model...")
            model = LogisticRegression()
        app_logger.info('Training Logisitc Model ...')
        model.fit(X_train, y_train)
        joblib.dump(model,model_path)
        predictions = model.predict(X_test)
        app_logger.info('Training Done!')
        return predictions
    
    def train_Random(self,X_train,y_train,X_test):
        '''
            This function is to train Logistic model
            params:
                X_train,X_test:array of vectorized text
                y_train: array of [0,1]
            return:
                predictions:array of {0,1}
        '''
        model_path = f"{worksapce}/Models/random.pkl"
        if os.path.exists(model_path):
            app_logger.info("Loading Existed Random Forest Model...")
            model = joblib.load(model_path)
        else:
            app_logger.info("Initilise Random Forest  Model...")
            model = RandomForestClassifier(n_estimators=500,max_depth=70,n_jobs=-1)
        app_logger.info('Training Random Forest Model ...')
        model.fit(X_train, y_train)
        joblib.dump(model,model_path)
        predictions = model.predict(X_test)
        app_logger.info('Training Done!')
        return predictions
    
    def train_XGB(self,X_train,y_train,X_test):
        '''
            This function is to train Logistic model
            params:
                X_train,X_test:array of vectorized text
                y_train: array of [0,1]
            return:
                predictions:array of {0,1}
        '''
        model_path = f"{worksapce}/Models/xgboost.json"
        if os.path.exists(model_path):
            model =XGBClassifier()
            app_logger.info("Loading Existed XGBoost Model...")
            model.load_model(model_path)
        else:
            app_logger.info("Initilise XGBoost Model...")
            model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, objective='binary:logistic')
        app_logger.info('Training XGBoost Model ...')
        model.fit(X_train, y_train)
        model.save_model(model_path)
        predictions =model.predict(X_test)
        app_logger.info('Training Done!')
        return predictions
    
    def train_model(self,model_name,df,X_train,y_train,X_test):
        if not isinstance(model_name,str):
            raise TypeError('Model Name is not a string')
        method = getattr(self,f'train_{model_name}',None)
        if callable(method):
            if model_name == 'VADER':
                prediction = method(df)
            else:
                prediction = method(X_train,y_train,X_test)
            return prediction
        else:
            raise Exception(f"Model '{model_name}' training process isn't defined please defint it then try again!")
        
if __name__ == '__main__':
    trainer = train()
    X_train = joblib.load(f'{worksapce}/Data/Vectorized/X_train_tfidf.pkl')
    X_test = joblib.load(f'{worksapce}/Data/Vectorized/X_test_tfidf.pkl')
    y_train = joblib.load(f'{worksapce}/Data/Vectorized/y_train.pkl')
    y_test = joblib.load(f'{worksapce}/Data/Vectorized/y_test.pkl')
    predictions = trainer.train_model('Logistic',X_train,y_train,X_test)


