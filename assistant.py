import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq_df = pd.read_csv('bank_faqs.csv')

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(faq_df['question'])

def get_response(user_input):
    user_tfidf = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    index = similarities.argmax()
    return faq_df['answer'][index]
while True:
    user_query = input("You: ")
    if user_query.lower() in ['exit', 'quit', 'bye']:
        print("Chatbot: Goodbye!")
        break
    response = get_response(user_query)
    print("Chatbot:", response)
