import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

print("Health & Wellness FAQ Chatbot (Simple Version)")
print("Type 'exit' to quit\n")

# Simple tokenizer function
def simple_tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())

# Load FAQs
def load_faqs(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except:
        print("Using sample FAQs instead of file")
        return [
            {"question": "How much water should I drink?", 
             "answer": "8-10 glasses (2-3 liters) daily"},
            {"question": "Symptoms of dehydration?", 
             "answer": "Thirst, dark urine, fatigue, dizziness"},
            {"question": "Recommended sleep duration?", 
             "answer": "7-9 hours per night for adults"},
            {"question": "What is a balanced diet?", 
             "answer": "Fruits, vegetables, lean proteins, whole grains"},
            {"question": "Exercise recommendations?", 
             "answer": "150 mins moderate or 75 mins vigorous weekly"}
        ]

# Load FAQs
faqs = load_faqs('faqs.json')
questions = [q['question'] for q in faqs]
answers = [q['answer'] for q in faqs]

# Create vectorizer
vectorizer = TfidfVectorizer(tokenizer=simple_tokenizer)
tfidf_matrix = vectorizer.fit_transform(questions)

# Chatbot function
def get_response(user_query):
    query_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    max_index = np.argmax(similarities)
    
    if similarities[0, max_index] > 0.3:
        return answers[max_index]
    return "I'm not sure. Can you ask differently?"

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
        
    response = get_response(user_input)
    print(f"Bot: {response}\n")