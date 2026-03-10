import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page Title
st.title("📧 Email Spam Classifier")

# 1. Training Logic (Direct-ah UI-la vekkalaam chinna project-naala)
data = {
    'email': [
        "Congratulations you won a lottery", "Meeting at 10 am tomorrow",
        "Get free money now", "Project submission tomorrow",
        "Win a free iPhone now", "Let's have lunch today"
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}
df = pd.DataFrame(data)
df['label'] = df['label'].map({'spam':1, 'ham':0})

# Model Setup
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['email'])
model = MultinomialNB()
model.fit(X, df['label'])

# 2. UI for User Input
user_msg = st.text_area("Enter Email message to check:")

if st.button("Predict"):
    if user_msg:
        vec = vectorizer.transform([user_msg])
        res = model.predict(vec)
        if res[0] == 1:
            st.error("🚨 This is a SPAM Email!")
        else:
            st.success("✅ This is NOT a Spam Email (Ham).")
    else:
        st.warning("Please type something!")
