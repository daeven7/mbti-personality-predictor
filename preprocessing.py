
import pandas as pd 
import numpy as np
from collections import Counter

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#%matplotlib inline
#import seaborn as sns
#sns.set_style("whitegrid")
#import scipy.stats as stats

import re
import string



import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import pickle


#from statsmodels.stats.multicomp import pairwise_tukeyhsd

#from textblob import TextBlob
   
df = pd.read_csv('mbti_1.csv') 
#print(df)


nltk.download('stopwords')

# Add to the stopwords list each of the 16 codes
types = ['infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp', 'isfp', 
         'istp', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj']
stop = stopwords.words('english')

for type in types:
    stop.append(type)

stop_rev = stop    
#print(stop_rev)

# Define a preprocessor function to clean the text by grouping into stems, removing separators, 
# replacing hyperlinks, removing punctuation, removing digits, and convert letters to lower case

def cleaner(text):
    stemmer = PorterStemmer()                                        # groups words having the same stems
    text = text.replace('|||', ' ')                                  # replaces post separators with empty space
    text = re.sub(r'\bhttps?:\/\/.*?[\r\n]*? ', 'URL ', text, flags=re.MULTILINE)  # replace hyperlink with 'URL'
    text = text.translate(str.maketrans('', '', string.punctuation)) # removes punctuation
    text = text.translate(str.maketrans('', '', string.digits))      # removes digits
    text = text.lower().strip()                                      # convert to lower case
    final_text = []
    for w in text.split():
        if w not in stop:
            final_text.append(stemmer.stem(w.strip()))
    return ' '.join(final_text)




# Create pipeline for the data preprocessing steps (CountVectorizer, TruncatedSVD) on the X data
pipeline_preprocessing2 = make_pipeline(
    CountVectorizer(preprocessor=cleaner, stop_words=stop_rev, ngram_range=(1,2), max_features=100))

pickle.dump(pipeline_preprocessing2, open('pipeline.pkl', 'wb'))
#load_pipeline = pickle.load(open(file_pipeline,'rb'))