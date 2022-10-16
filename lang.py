import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re

cv = CountVectorizer()
le = LabelEncoder()

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb')) instead of pickle file we have used the code

@app.route('/')
def home():
    return render_template('language detection.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    df = pd.read_csv(".\Language Detection.csv")

    X = df["Text"]
    y = df["Language"]

    y = le.fit_transform(y)

    text_list = []

    # iterating through all the text
    for text in X:         
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text) # removes all the symbols and numbers
        text = re.sub(r'[[]]', ' ', text)   
        text = text.lower()          # converts all the text to lower case
        text_list.append(text)       # appends the text to the text_list
    
    X = cv.fit_transform(text_list).toarray() 
                                            
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.80)

    
    from sklearn.naive_bayes import MultinomialNB  
    model = MultinomialNB()
    model.fit(x_train, y_train)

    if request.method == 'POST':
        txt = request.form['text']
        x = cv.transform([txt]).toarray()# convert text to bag of words model (Vector)
        language = model.predict(x) # predict the language
        lang = le.inverse_transform(language) # find the language corresponding with the predicted value
    
        output = lang[0]

    return render_template('language detection.html', prediction='Language is in {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)