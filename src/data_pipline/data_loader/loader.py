import os 
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd
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
        '''
            This function is to save the cleaned data frame as CSV file
            params:
                df: data frame
        '''
        desired_columns = ['cleaned_text','cleaned_tokens','sentiment']
        df = df[desired_columns]
        df['label'] = df['sentiment'].map({'POSITIVE': 1, 'NEGATIVE': 0})
        print(df.isnull().sum())
        df.to_csv(f'{worksapce}/Data/Cleaned dataset/cleaned_news.csv',index=False)
        app_logger.info("Save cleaned data is done successfully!")
        return df
    
    def vectorize_data(self,text,save_name):
        '''
            This function is to vectorize text.
            params:
                text: can be a data frame or a string
                save_name: string the name of saved vectorized file
            return:
                vectorized_text
        '''
        vectorizer_path = f'{worksapce}/Models/Vectorizer/victorizer.pkl'
        if os.path.exists(vectorizer_path):
            app_logger.info("Loading existing Vectorizer ...")
            vectorizer = joblib.load(vectorizer_path)
            text_tfidf = vectorizer.transform(text)
        else:
            app_logger.info("Initialize a new vectorizer ...")
            vectorizer = TfidfVectorizer()
            text_tfidf = vectorizer.fit_transform(text)
            joblib.dump(vectorizer, f'{worksapce}/Models/Vectorizer/victorizer.pkl')
        if text_tfidf == None:
            raise Exception('Exeption: Unexpected Error accured')
        app_logger.info(f'{save_name} is vectorized successfully!')
        joblib.dump(text_tfidf, f'{worksapce}/Data/Vectorized/{save_name}_tfidf.pkl')
        app_logger.info(f'vectorized data is saved successfully!')
        return text_tfidf
