import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy.sparse import csr_matrix

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Set NLTK data directory (Ensure the directory exists and has necessary data)
nltk.data.path.append("./nltk_data")

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]

    y = [
        i
        for i in y
        if i not in stopwords.words("english") and i not in string.punctuation
    ]

    y = [ps.stem(i) for i in y]

    return " ".join(y)


# Loading vectorizer and model
with open("vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

# Sidebar sample messages
st.sidebar.title("Sample Messages")
sample_messages = [
    "1. Go to Jurong Point. It's crazy! Available only in Bugis and Great World: the buffet. Cine there has Amore, right?",
    "2. Free entry into a weekly competition to win FA Cup final tickets for the 21st of May, 2005. Text 'FA' to 87121 to receive the entry question (standard text rate). Terms and conditions apply. Contact 08452810075 for over 18's.",
    "3. SIX chances to win CASH! From £100 to £20,000. Text 'CSH11' and send to 87575. Cost: 150p/day, 6 days, 16+. Terms and conditions apply. Reply 'HL' for info.",
    "4. After I finish my lunch, then I'll head straight down. Around 3, perhaps. Have you finished your lunch already?",
]

# Initialize an empty message variable
selected_message = ""

# Display checkboxes for sample messages
for msg in sample_messages:
    if st.sidebar.checkbox(msg):
        selected_message = msg.split(" ", 1)[1]

# If a checkbox was selected, use its message
if selected_message:
    input_sms = selected_message
    st.text_area("Message from Selected Text", input_sms)

if st.button("Predict"):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
