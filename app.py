import streamlit as st
import pickle 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
   
    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    ps = PorterStemmer()
    for word in text:
        if word not in stopwords.words('english'):
            y.append(ps.stem(word))

    return " ".join(y)

vectorizer = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")

input_msg = st.text_area("Enter your email")

if st.button("Predict"):

    #preprocess
    transform_msm = transform_text(input_msg)

    #vectorize
    vector_input = vectorizer.transform([transform_msm])

    #predict
    result = model.predict(vector_input)[0]

    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")





