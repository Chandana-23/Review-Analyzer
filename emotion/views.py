import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline



from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request,'index.html')

def detect(request):
    if request.method=="POST":
        review = request.POST["review"]
        df=pd.read_table(r"templates\Restaurant_Reviews (2).csv") #creating a dataframe(2-d data)
        x=df['Review'].values #x is input
        y=df['Liked'].values  # is output that we have to predict
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0) #divided into 4 datasets
        vect=CountVectorizer(stop_words='english')  #transforms the text into a vector
        x_train_vect=vect.fit_transform(x_train)
        x_test_vect=vect.transform(x_test)
        model=SVC()
        model.fit(x_train_vect,y_train)
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf',TfidfTransformer()),
            ('clf', SGDClassifier()),
        ])
        predicted = pipeline.fit(x_train,y_train)
        # Now evaluate all steps on test set
        predicted = pipeline.predict(x_test)
        text_model=make_pipeline(CountVectorizer(),SVC()) #use the pipeline method
        text_model.fit(x_train,y_train) #again training the model
        n = text_model.predict([review])
        print(n[0])
        return render(request,'result.html',{'n':n})
    return render(request,'index.html')

