# tf_idf_vector representation of text(sentence) with n_gram_bag of_words(bigram) without using library 

import math
from collections import Counter
import re
from nltk.tokenize import word_tokenize
import argparse
import os

def pre_process(text):
    """Preprocess the text by tokenizing and generating bigrams."""
    text = re.sub(r'[\n\t]', ' ', text)          # Remove newlines and tabs
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and convert to lowercase
    tokens = word_tokenize(text)                # Tokenize the text
    
    # Generate bigrams
    n_grams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    return tokens + n_grams

def calculate_tf(doc):
    """Calculate Term Frequency (TF) for a document."""
    term_freq = Counter(doc)
    max_frequency = max(term_freq.values()) if term_freq else 1
    tf = {term: 0.5 + 0.5 * (count / max_frequency) for term, count in term_freq.items()}
    return tf

def calculate_idf(docs):
    """Calculate Inverse Document Frequency (IDF) for all terms."""
    total_docs = len(docs)
    term_doc_freq = {}
    for doc in docs:
        unique_terms = set(doc)
        for term in unique_terms:
            term_doc_freq[term] = term_doc_freq.get(term, 0) + 1
    idf = {term: math.log(total_docs / (1 + doc_freq)) for term, doc_freq in term_doc_freq.items()}
    return idf

def calculate_tfidf(docs):
    """Calculate TF-IDF for each document."""
    tf = [calculate_tf(doc) for doc in docs]
    idf = calculate_idf(docs)
    tfidf = [{term: doc_tf.get(term, 0) * idf.get(term, 0) for term in doc_tf} for doc_tf in tf]
    return tfidf

def main():
    """Main function to compute n-gram Bag-of-Words with TF-IDF."""
    parser = argparse.ArgumentParser(description="Compute n-gram Bag-of-Words with TF-IDF.")
    parser.add_argument("input_file", type=str, help="Path to input text file.")
    args = parser.parse_args()
    
    # Ensure input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: File '{args.input_file}' does not exist.")
        return
    
    # Read input file
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
        # Preprocess sentences
        cleaned_sentences = [pre_process(line.strip()) for line in lines if line.strip()]
        
        # Build vocabulary
        vocabulary = set()
        for sentence in cleaned_sentences:
            vocabulary.update(sentence)
        vocabulary_list = sorted(vocabulary)             #  consistent order 
        
        # Calculate TF-IDF
        tfidf = calculate_tfidf(cleaned_sentences)
        
        # Create sentence vectors
        sentence_vectors = []
        for sentence_tfidf in tfidf:
            vector = [sentence_tfidf.get(term, 0) for term in vocabulary_list]
            sentence_vectors.append(vector)
    
    # Print results
    print("Vocabulary:", vocabulary_list)
    print("Final Vectors:")
    for vector in sentence_vectors:
        print(vector)

if __name__ == "__main__":
    main()
