import os
import sys
from sklearn.model_selection import train_test_split
import joblib
# Add the directory containing the `src` folder to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))  # Adjust to reach `src`
sys.path.append(PROJECT_ROOT)
current_path = os.getcwd()
worksapce = current_path.split('src')[0]

from utils.logging_utils.app_logger import app_logger
from data_pipline.data_extractor.extractor import LoadCSVFile
from data_pipline.data_transformer.transformer import transform
from data_pipline.data_loader.loader import Load
class data_pipeline:

    def apply(self,path,mode):
        try:
            if not isinstance(path,str):
                raise TypeError('Path is not a string type')
            extractor = LoadCSVFile()
            df = extractor.load_data(path)
            transformer = transform()
            df = transformer.clean_data(df)
            iloader = Load()
            if mode == 'train':
                df = iloader.save_data(df)
                app_logger.info('Spliting data to train,test,val..')
                X_train, X_temp, y_train, y_temp = train_test_split(df['cleaned_text'], df['label'], test_size=0.4, random_state=42)
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
                X_train_tfidf = iloader.vectorize_data(X_train,'X_train')
                X_val_tfidf = iloader.vectorize_data(X_val,'X_val')
                X_test_tfidf = iloader.vectorize_data(X_test,'X_test')
                joblib.dump(y_train,f'{worksapce}/Data/Vectorized/y_train.pkl')
                joblib.dump(y_test,f'{worksapce}/Data/Vectorized/y_test.pkl')
                joblib.dump(y_val,f'{worksapce}/Data/Vectorized/y_val.pkl')
                return df,X_train_tfidf,X_val_tfidf,X_test_tfidf
            else:
                vector = iloader.vectorize_data(df['cleaned_text'],'temp')
                return df,vector
        except Exception as e:
            app_logger.error(f'Exeption:{e}')

if __name__ == '__main__':
    
    pipline = data_pipeline()
    pipline.apply('C:/Users/USER/Desktop/Qufzah/project/Data/Source_dataset/news.csv')

