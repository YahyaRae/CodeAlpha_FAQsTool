import json
import nltk
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

print("Initializing Health & Wellness Chatbot...")

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Initialize stemmer
stemmer = PorterStemmer()

def load_faqs(file_path):
    try:
        print(f"Loading FAQs from {file_path}...")
        with open(file_path, 'r') as file:
            faqs = json.load(file)
            print(f"Successfully loaded {len(faqs)} FAQs")
            return faqs
    except Exception as e:
        print(f"Error loading FAQs: {e}")
        return []

def preprocess(text):
    try:
        tokens = nltk.word_tokenize(text.lower())
        return " ".join([stemmer.stem(word) for word in tokens if word.isalnum()])
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return text.lower()

class FAQChatbot:
    def __init__(self, faqs):
        print("Training chatbot...")
        self.faqs = faqs
        self.vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
        
        # Preprocess questions
        questions = [preprocess(faq['question']) for faq in self.faqs]
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(questions)
        print("Chatbot training complete!")
    
    def get_response(self, user_query):
        try:
            processed_query = preprocess(user_query)
            query_vector = self.vectorizer.transform([processed_query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)
            max_index = np.argmax(similarities)
            
            if similarities[0, max_index] > 0.3:
                return self.faqs[max_index]['answer']
            return "I'm not sure about that. Could you rephrase your question?"
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I encountered an error processing your question"

# Initialize chatbot
print("Starting chatbot setup...")
faqs = load_faqs('faqs.json')
if not faqs:
    print("No FAQs loaded. Using sample FAQs instead.")
    faqs = [
        {"question": "How much water should I drink daily?", 
         "answer": "Adults should drink 8-10 glasses (2-3 liters) of water daily."},
        {"question": "What are symptoms of dehydration?", 
         "answer": "Common signs include thirst, dark urine, fatigue, dizziness."}
    ]

chatbot = FAQChatbot(faqs)
print("\nHealth & Wellness FAQ Chatbot is ready!")
print("Type your health questions or 'exit' to quit\n")

# Chat loop
while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        response = chatbot.get_response(user_input)
        print(f"Bot: {response}\n")
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")