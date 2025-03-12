import emoji
import os
import sys
import re
from nltk.corpus import stopwords
import string
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt_tab') 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

class preprocessings:
    '''
        This class contains all text preprocessing methods 
        feel free to add new ones by declaring a new function as follow:
        def function_name(self,text):
            # TODO
            retunn text
    '''

    def lowecasing(self,text):
        '''
            This function is to lowercase text
            params:
                text:string
            return:
                text.lower():string
        '''
        return text.lower()
    

    def check_mentions(self,text):
        '''
            This function is to check if text contains mentions such as @username
            params:
                text:string
            return:
                mentions:list[string]
        '''
        mentions = re.findall(r'@\w+', text)
        return mentions
    

    def remove_mentions(self,text):
        '''
            This function is to remove mentions such as @username from given string
            params:
                text:string
            return:
                text:string
        '''
        return re.sub(r'@\w+', '', text)
    

    def check_urls(self,text):
        '''
            This function is to check if text contains urls such as www.example.com ,etc
            params:
                text:string
            return:
                urls:list[string]
        '''
        urls = re.findall(r'http\S+|www.\S+', text)
        return urls
    
    def remove_urls(self,text):
        '''
            This function is to remove mentions such as www.example.com from given string
            params:
                text:string
            return:
                text:string
        '''
        return re.sub(r'http\S+|www.\S+', '', text)
    

    def check_hashtags(self,text):
        '''
            This function is to check if text contains hashtags such as #hashtag
            params:
                text:string
            return:
                hashtags:list[string]
        '''
        hashtags = re.findall(r'#\w+', text)
        return hashtags
    
    def remove_hashtags(self,text):
        '''
            This function is to remove hashtags such as #hashtag from given string
            params:
                text:string
            return:
                text:string
        '''
        return re.sub(r'#\w+', '', text)
    

    def check_emoji(self,text):
        '''
            This function is to check if text contains emojies
            params:
                text:string
            return:
                var:boolean
        '''
        return any(char in emoji.EMOJI_DATA for char in text)
    
    def replace_emojis(text):
        '''
            This function is to replace empjies with strings
            params:
                text:string
            return:
                text:string
        '''
        return emoji.demojize(text, delimiters=(" ", " "))
    

    def remove_stopwords(self,tokens):
        '''
            This function is to remove stopwords
            params:
                tokens:list[string]
            return:
                text:string
        '''
        stop_words = set(stopwords.words('english'))
        cleaned_tokens = [word for word in tokens if word.lower() not in stop_words ]
        return ' '.join(cleaned_tokens)
    

    def remove_puncutations(self,tokens):
        '''
            This function is to remove puncutations
            params:
                tokens:list[string]
            return:
                text:string
        '''
        punctuation = set(string.punctuation)
        cleaned_tokens = [word for word in tokens if word.lower() not in punctuation ]
        return ''.join(cleaned_tokens)
    

    def get_wordnet_pos(self,tag):
        """
            Convert POS tags from NLTK to WordNet format
            params:
                tag:string
            return:
                wordnet pos tag
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN 
        

    def lemmatize_sentence(self,tokens):
        '''
            This function is to lemmatize text such converting running to run
            params:
                tokens:list[string]
            return:
                text:string
        '''
        lemmatizer = WordNetLemmatizer()
        pos_tags = pos_tag(tokens)  # Part-of-speech tagging
        lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
        return ' '.join(lemmatized_words)