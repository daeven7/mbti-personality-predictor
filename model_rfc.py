
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

import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler


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

df_working = df.copy()



# Create a binary column for each of the 4 axis types for later analysis
df_working['I-E'] = df_working['type'].map(lambda x: 'Introversion' if x[0] == 'I' else 'Extroversion')
df_working['N-S'] = df_working['type'].map(lambda x: 'Intuition' if x[1] == 'N' else 'Sensing')
df_working['T-F'] = df_working['type'].map(lambda x: 'Thinking' if x[2] == 'T' else 'Feeling')
df_working['J-P'] = df_working['type'].map(lambda x: 'Judging' if x[3] == 'J' else 'Perceiving')

#df_working.head()


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


# Determine baseline for each of the four axes
baseline_IE = df_working['I-E'].value_counts().max() / df_working['I-E'].value_counts().sum()
baseline_NS = df_working['N-S'].value_counts().max() / df_working['N-S'].value_counts().sum()
baseline_TF = df_working['T-F'].value_counts().max() / df_working['T-F'].value_counts().sum()
baseline_JP = df_working['J-P'].value_counts().max() / df_working['J-P'].value_counts().sum()


# Train-test splits, using type variables as target and posts variable as predictor
# Introversion - Extroversion
X_train_IE, X_test_IE, y_train_IE, y_test_IE = train_test_split(df_working['posts'].values,
                                                   df_working['I-E'].values,
                                                   test_size=0.30, random_state=42)
# Intuition - Sensing
X_train_NS, X_test_NS, y_train_NS, y_test_NS = train_test_split(df_working['posts'].values,
                                                   df_working['N-S'].values,
                                                   test_size=0.30, random_state=42)
# Thinking - Feeling
X_train_TF, X_test_TF, y_train_TF, y_test_TF = train_test_split(df_working['posts'].values,
                                                   df_working['T-F'].values,
                                                   test_size=0.30, random_state=42)
# Judging - Perceiving
X_train_JP, X_test_JP, y_train_JP, y_test_JP = train_test_split(df_working['posts'].values,
                                                   df_working['J-P'].values,
                                                   test_size=0.30, random_state=42)

# Create pipeline for the data preprocessing steps (CountVectorizer, TruncatedSVD) on the X data
pipeline_preprocessing2 = make_pipeline(
    CountVectorizer(preprocessor=cleaner, stop_words=stop_rev, ngram_range=(1,2), max_features=100))

# Preprocess each of the train-test splits
# Introversion - Extroversion
X_train_IE_tsvd = pipeline_preprocessing2.fit_transform(X_train_IE)
X_test_IE_tsvd = pipeline_preprocessing2.transform(X_test_IE)

# Intuition - Sensing
X_train_NS_tsvd = pipeline_preprocessing2.fit_transform(X_train_NS)
X_test_NS_tsvd = pipeline_preprocessing2.transform(X_test_NS)

# Thinking - Feeling
X_train_TF_tsvd = pipeline_preprocessing2.fit_transform(X_train_TF)
X_test_TF_tsvd = pipeline_preprocessing2.transform(X_test_TF)

# Judging - Perceiving
X_train_JP_tsvd = pipeline_preprocessing2.fit_transform(X_train_JP)
X_test_JP_tsvd = pipeline_preprocessing2.transform(X_test_JP)




oversample = RandomOverSampler(sampling_strategy='minority')

Xsm_train_IE_tsvd, ysm_train_IE = oversample.fit_resample(X_train_IE_tsvd, y_train_IE)


# import random search, random forest, iris data, and distributions
from sklearn.model_selection import RandomizedSearchCV
#from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform, truncnorm, randint



rfc_IE = RandomForestClassifier(min_samples_leaf=1, min_samples_split=0.0136, n_estimators=170, 
                             criterion='gini',max_features=0.26, bootstrap='True', n_jobs= -1, random_state=123)
rfc_IE.fit(Xsm_train_IE_tsvd, ysm_train_IE)



oversample = RandomOverSampler(sampling_strategy='minority')

Xsm_train_JP_tsvd, ysm_train_JP = oversample.fit_resample(X_train_JP_tsvd, y_train_JP)

rfc_JP = RandomForestClassifier(min_samples_leaf=1, min_samples_split=0.0131, n_estimators=116, 
                             criterion='gini',max_features=0.25, bootstrap='True', n_jobs= -1, random_state=123)
rfc_JP.fit(Xsm_train_JP_tsvd, ysm_train_JP)




oversample = RandomOverSampler(sampling_strategy='minority')

Xsm_train_NS_tsvd, ysm_train_NS = oversample.fit_resample(X_train_NS_tsvd, y_train_NS)


rfc_NS = RandomForestClassifier(min_samples_leaf=1, min_samples_split=0.0131, n_estimators=116, 
                             criterion='gini',max_features=0.25, bootstrap='True', n_jobs= -1, random_state=123)
rfc_NS.fit(Xsm_train_NS_tsvd, ysm_train_NS)


rfc_TF = RandomForestClassifier(min_samples_leaf=1, min_samples_split=0.0131, n_estimators=116, 
                             criterion='gini',max_features=0.25, bootstrap='True', n_jobs= -1, random_state=123)
rfc_TF.fit(X_train_TF_tsvd, y_train_TF)


file_IE = 'rfc_IE.pkl'
file_JP = 'rfc_JP.pkl'
file_NS = 'rfc_NS.pkl'
file_TF = 'rfc_TF.pkl'
file_pipeline='pipeline.pkl'

pickle.dump(rfc_IE, open(file_IE, 'wb'))
pickle.dump(rfc_JP, open(file_JP, 'wb'))
pickle.dump(rfc_NS, open(file_NS, 'wb'))
pickle.dump(rfc_TF, open(file_TF, 'wb'))
pickle.dump(rfc_TF, open(file_TF, 'wb'))
pickle.dump(pipeline_preprocessing2, open(file_pipeline, 'wb'))



def predict(str):
  str=[str]

  load_model_rfc_IE = pickle.load(open(file_IE, 'rb'))
  load_model_rfc_JP = pickle.load(open(file_JP, 'rb'))
  load_model_rfc_NS = pickle.load(open(file_NS, 'rb'))
  load_model_rfc_TF = pickle.load(open(file_TF, 'rb'))
  load_pipeline = pickle.load(open(file_pipeline,'rb'))

  test_input = load_pipeline.fit_transform(str)
  IE=load_model_rfc_IE.predict(test_input)[0]
  JP=load_model_rfc_JP.predict(test_input)[0]
  NS=load_model_rfc_NS.predict(test_input)[0]
  TF=load_model_rfc_TF.predict(test_input)[0]

  res= IE+" "+JP+" "+NS+" "+" "+TF
  return res


#str1="""I worked till 2014 with wonderful boss and company for 11 years . Whenever you talk with even closest friend about retirement ,do not tell you have enough money to support you . Because many more people are waiting to take your position .If you change your mind ,they already know that you need to retire . That was the best conversation I had with my boss and somebody else took my position and boss would always miss my good work . That is fine as if I need a new job I have good records with my boss as a good employee .thank you very much Always respect your boss and coworkers even they are not very friendly ."""
#predict(str1)

load_model_rfc_IE = pickle.load(open(file_IE, 'rb'))
load_model_rfc_JP = pickle.load(open(file_JP, 'rb'))
load_model_rfc_NS = pickle.load(open(file_NS, 'rb'))
load_model_rfc_TF = pickle.load(open(file_TF, 'rb'))
load_pipeline = pickle.load(open(file_pipeline,'rb'))