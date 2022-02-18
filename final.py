from re import T, X
from sqlite3 import Date, TimeFromTicks
import pandas as pd
from pandas import read_csv
import numpy as np
import pandas
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
email =pd. read_csv('emails.csv')
Data=email.head()
# print(Data)
# clean data
filter_vocab=[]
special_char=[",",":"," ",";",".","?"]
stopWords=["to","is","a"]
def stopWords(txt):
    txt = txt.split()  
    ntxt = ""
    for i in txt:
        if i not in stopwords:
           ntxt += i+" "
    return(ntxt.strip())  
def cleanData(Data,textLabel):
    Data[textLabel]=Data[textLabel].apply(lambda x: x.lower())
    Data[textLabel]=Data[textLabel].apply(lambda x: r.sub("[^a-z0-9\s","",x))
    Data[textLabel]=Data[textLabel].apply(lambda x: r.sub("\s+"," ",),x)
    Data[textLabel]=Data[textLabel].apply(lambda x: x.stopWords(x))
# feature extruction
def features(DB,col):
    vectorizer = TfidfVectorizer()
    feature=vectorizer.fit (DB,[col])  
featurs=features(Data,'text')
x_train,x_test,y_train,y_test=train_test_split(Data["spam"],featurs,test_size=0.2,random_state=1)
model=LogisticRegression()
model.fit(x_train,y_train)
prdict=model.predict(x_test)
print(metrics.accuracy_score(y_test,prdict)*100)
print(metrics.classification_report(y_test,prdict))
# #finding unique
unique = []
def unidue():
    for word in Data:
        if word not in unique:
            unique.append(word)
#sort
spam=unique.sort()
#print
print(unique.list())
notSpam=unique.copy()
feat=spam+notSpam
print(feat.list())





















    

    

# unique = []
# for word in words:
#     if word not in unique:
#         unique.append(word)
# unique.sort()
# print(unique)