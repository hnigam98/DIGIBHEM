import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
data_fake = pd.read_csv('fake.csv')
data_true = pd.read_csv('true.csv')
data_fake.head()
data_true.head()
data_fake["class"] = 0
data_true['class'] = 1
data_fake.shape, data_true.shape
manual_testing = data_fake.tail(10)
manual_testing = data_true.tail(10)
for i in range(23480,23470,1):
    data_fake.drop([i], axis=0, inplace=True)
   
for i in range(21416,21406,1):
    data_fake.drop([i], axis=0, inplace=True)
data_fake['class']=0
data_true['class']=1
data_merge=pd.concat([data_fake, data_true],axis=0)
data_merge.head(10)
data = data_merge.drop(['title','subject','date'],axis=1)
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\w"," ",text)
    text = re.sub('https?://s+|www\.\s+','',text)
    text = re.sub('<.*?>+',' ',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    return text
data['text']=data['text'];
x =data['text']
y =data['class']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
x_train = x_train.astype(str)
xv_train = vectorization.fit_transform(x_train)
x_test = x_test.astype(str)
xv_test = vectorization.transform(x_test)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xv_train ,  y_train)
LR.score,(xv_test,y_test)
pred_LR = LR.predict(xv_test)
print(classification_report(y_test,pred_LR))
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(xv_train,  y_train)
pred_dt = DT.predict(xv_test)
DT.score(xv_test, y_test)
print(classification_report(y_test, pred_dt))
def output_label(n):
    if n == 0:
        return  "Fake news"
    elif n == 1:
        return "Not a Fake news"
    manual_testing_data = data_true.tail(10)
def manual_testing(news):
        testing_news = {"text": [news]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["text"] = new_def_test["text"].apply(wordopt)
        new_x_test = new_def_test["text"]
        new_xv_test = vectorization.transform(new_x_test)
        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test) 
        
        print("\n\nLR prediction: {} \nDT prediction: {}".format(output_label(pred_LR[0]), output_label(pred_DT[0])))
                 
news = "This is a test news article for manual testing."
manual_testing(news) 
