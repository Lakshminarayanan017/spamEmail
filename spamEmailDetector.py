import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


#page configuration


st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="🤖",
    layout = "centered"
)

st.subheader("💬 Try it Yourself!")
user_input = st.text_area("Enter an Email Message to classify: ")

#load and split train and test Data

df = pd.read_csv("email_dataset.csv")
X = df["text"]
y = df["label_num"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

#Vectorizing

vector = vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4))
X_train_vec = vector.fit_transform(X_train)
X_test_vec = vector.transform(X_test)

#Model

model = LogisticRegression()
model.fit(X_train_vec,y_train)
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_pred,y_test)

#Predict on user input

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        input_df = pd.DataFrame({"text" : [user_input]})
        input_vec = vector.transform(input_df["text"])
        prediction = model.predict(input_vec)[0]
        label = ""
        if prediction == 0:
            label = "📮 Geniune Message"
        elif prediction == 1:
            label = "🚫 Spam"
        else:
            label = "👨🏻‍💼 Advertisement"
        st.success(f"Prediction: **{label}**")
        
        
