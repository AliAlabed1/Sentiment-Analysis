import os 
import sys
from nltk.tokenize import word_tokenize
import re
import ast


# Add the directory containing the `src` folder to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))  # Adjust to reach `src`
sys.path.append(PROJECT_ROOT)

from utils.logging_utils.app_logger import app_logger
from utils.preprocessings_utils.preprocessings import preprocessor

class transform:
    '''
        This class is to transform and clean data
    '''

    def simple_tokenize(self,text):
        """
        A simple tokenizer using NLTK word_tokenize.
        You could replace this with a more sophisticated approach
        (e.g., spaCy) if desired.
        """
        # Lowercase
        text = text.lower()
        # Remove extra characters (optional)
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        # Tokenize
        tokens = word_tokenize(text)
        return tokens

    def clean_data(self,df):
        '''
            This function is to clean text data 
            params:
                df: data frame 
            return:
                result: data frame
        '''
        ipreprocessor = preprocessor()
        app_logger.info("Start cleaning data...")

        df['news'] = df['news'].apply(lambda x: ast.literal_eval(x).decode('utf-8') if isinstance(x, str) and (x.startswith("b'") or x.startswith('b"')) else x)


        # check for duplicates
        app_logger.info('Checking for Duplicates')
        num_duplicates = df['news'].duplicated().sum()
        if num_duplicates >0:
            app_logger.info(f'Ther is {num_duplicates} duplicated sample')
            df.drop_duplicates(subset='news', inplace=True)
            app_logger.info('Remove Duplicates successfully')
        else:
            app_logger.info("There is no dupllicated rows ")

        # Lower casing
        app_logger.info("Lower casing..")
        df['cleaned_text'] = df['news'].apply(lambda x: ipreprocessor.apply_method('lowercasing', x))
        app_logger.info('Lower casing Applied Successfully')

        # checking and remove mentions
        app_logger.info("Checking for Mentions...")
        df['mentions'] = df['cleaned_text'].apply(lambda x: ipreprocessor.apply_method('check_mentions', x))
        contain_mentions = len(df[df['mentions'].apply(len)>0])
        if contain_mentions > 0:
            app_logger.info(f'{contain_mentions} samples contain mentions')
            app_logger.info("Removing Mentions ...")
            df.loc[df['mentions'].apply(len) > 0, 'cleaned_text'] = df.loc[df['mentions'].apply(len) > 0, 'cleaned_text'].apply(lambda x: ipreprocessor.apply_method('remove_mentions', x))
            df['mentions'] = df['cleaned_text'].apply(lambda x: ipreprocessor.apply_method('check_mentions', x))
            contain_mentions = len(df[df['mentions'].apply(len)>0])
            if contain_mentions == 0:
                app_logger.info('Removing Mentions applied successfully')
            else:
                raise Exception('Unexpected error accured while removing mentions')
            
        # checking and remove urls
        app_logger.info("Checking for URLS...")
        df['urls'] = df['cleaned_text'].apply(lambda x: ipreprocessor.apply_method('check_urls', x))
        contain_urls = len(df[df['urls'].apply(len)>0])
        if contain_urls > 0:
            app_logger.info(f'{contain_urls} samples contain urls')
            app_logger.info("Removing URLs ...")
            df.loc[df['urls'].apply(len) > 0, 'cleaned_text'] = df.loc[df['urls'].apply(len) > 0, 'cleaned_text'].apply(lambda x: ipreprocessor.apply_method('remove_urls', x))
            df['urls'] = df['cleaned_text'].apply(lambda x: ipreprocessor.apply_method('check_urls', x))
            contain_urls = len(df[df['urls'].apply(len)>0])
            if contain_urls == 0:
                app_logger.info('Removing URLs applied successfully')
            else:
                raise Exception('Unexpected error accured while removing URLs')
            
        # checking and remove hashtags
        app_logger.info("Checking for hashtags...")
        df['hashtags'] = df['cleaned_text'].apply(lambda x: ipreprocessor.apply_method('check_hashtags', x))
        contain_hashtags = len(df[df['hashtags'].apply(len)>0])
        if contain_hashtags > 0:
            app_logger.info(f'{contain_hashtags} samples contain hashtags')
            app_logger.info("Removing hashtags ...")
            df.loc[df['hashtags'].apply(len) > 0, 'cleaned_text'] = df.loc[df['hashtags'].apply(len) > 0, 'cleaned_text'].apply(lambda x: ipreprocessor.apply_method('remove_hashtags', x))
            df['hashtags'] = df['cleaned_text'].apply(lambda x: ipreprocessor.apply_method('check_hashtags', x))
            contain_hashtags = len(df[df['hashtags'].apply(len)>0])
            if contain_hashtags == 0:
                app_logger.info('Removing hashtags applied successfully')
            else:
                raise Exception('Unexpected error accured while removing hashtags')
        
        # checking and remove emojis
        app_logger.info("Checking for emojis...")
        df['emojis'] = df['cleaned_text'].apply(lambda x: ipreprocessor.apply_method('check_emoji', x))
        contain_emojis = len(df[df['emojis']])
        if contain_emojis > 0:
            app_logger.info(f'{contain_emojis} samples contain emojis')
            app_logger.info("Replacing emojis ...")
            df.loc[df['emojis'], 'cleaned_text'] = df.loc[df['emojis'], 'cleaned_text'].apply(lambda x: ipreprocessor.apply_method('replace_emojis', x))
            df['emojis'] = df['cleaned_text'].apply(lambda x: ipreprocessor.apply_method('check_emoji', x))
            contain_emojis = len(df[df['emojis']])
            if contain_emojis == 0:
                app_logger.info('Replacing emojis applied successfully')
            else:
                raise Exception('Unexpected error accured while Replacing emojis')
            

        # remove stopwords and puncutations
        app_logger.info("Removing stopwords and punctations...")
        app_logger.info("Tokenize the text...")
        df['cleaned_tokens'] = df['cleaned_text'].apply(self.simple_tokenize)
        df['cleaned_text'] = df['cleaned_tokens'].apply(lambda x: ipreprocessor.apply_method('remove_stopwords', x)).apply(lambda x: ipreprocessor.apply_method('remove_puncutations', x))
        app_logger.info("Removing puncutations and stopwords applied successfully!")
        


        # limmatize the text
        app_logger.info("Lemmatizing the text...")
        app_logger.info("Retokenize the text after removing stopwords...")
        df['cleaned_tokens'] = df['cleaned_text'].apply(self.simple_tokenize)
        df['cleaned_text'] = df['cleaned_tokens'].apply(lambda x: ipreprocessor.apply_method('lemmatize_sentence',x))
        app_logger.info("Lemmatize text applied successfully!")
        return df
        

if __name__ == '__main__':
    itransform = transform()
