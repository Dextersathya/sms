import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Set NLTK data directory
nltk.data.path.append("./nltk.txt")

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

# 1. Display first message
st.sidebar.title("Sample Messages")
msg_1 = "1. Go to Jurong Point. It's crazy! Available only in Bugis and Great World: the buffet. Cine there has Amore, right?"
if st.sidebar.checkbox(
    " Go to Jurong Point. It's crazy! Available only in Bugis and Great World: the buffet. Cine there has Amore, right?"
):
    input_sms = msg_1.split(" ", 1)[1]
    st.text_area("Message from Selected Text", input_sms)

# 2. Display second message
msg_2 = "2. Free entry into a weekly competition to win FA Cup final tickets for the 21st of May, 2005. Text 'FA' to 87121 to receive the entry question (standard text rate). Terms and conditions apply. Contact 08452810075 for over 18's."
if st.sidebar.checkbox(
    "2. Free entry into a weekly competition to win FA Cup final tickets for the 21st of May, 2005. Text 'FA' to 87121 to receive the entry question (standard text rate). Terms and conditions apply. Contact 08452810075 for over 18's."
):
    input_sms = msg_2.split(" ", 1)[1]
    st.text_area("Message from Selected Text", input_sms)

# 3. Display third message
msg_3 = "3. SIX chances to win CASH! From £100 to £20,000. Text 'CSH11' and send to 87575. Cost: 150p/day, 6 days, 16+. Terms and conditions apply. Reply 'HL' for info."
if st.sidebar.checkbox(
    "SIX chances to win CASH! From £100 to £20,000. Text 'CSH11' and send to 87575. Cost: 150p/day, 6 days, 16+. Terms and conditions apply. Reply 'HL' for info."
):
    input_sms = msg_3.split(" ", 1)[1]
    st.text_area("Message from Selected Text", input_sms)

# 4. Display fourth message
msg_4 = "4. After I finish my lunch, then I'll head straight down. Around 3, perhaps. Have you finished your lunch already?"
if st.sidebar.checkbox(
    "After I finish my lunch, then I'll head straight down. Around 3, perhaps. Have you finished your lunch already?"
):
    input_sms = msg_4.split(" ", 1)[1]
    st.text_area("Message from Selected Text", input_sms)

if st.button("Predict"):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
