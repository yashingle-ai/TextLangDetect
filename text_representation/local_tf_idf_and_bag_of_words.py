# Here we calculate TF-IDF for a specific range of sentences called local TF-IDF

import math
from collections import Counter  # Import Counter for frequency counting
import re
from nltk.tokenize import word_tokenize
import numpy as np


def pre_process(text):
    """
    Preprocess the input text by:
    - Removing newlines and tabs
    - Removing punctuation and converting to lowercase
    - Tokenizing text into words
    """
    text = re.sub(r'[\n\t]', ' ', text)             # Remove newlines and tabs
    text = re.sub(r'[^\w\s]', '', text.lower())     # Remove punctuation and convert to lowercase
    return word_tokenize(text) or []               # Tokenize text into words


# Function to calculate TF-IDF for a local range of sentences
# This function takes:
# - `sentences`: Tokenized sentences from the input file
# - `start` and `end`: Indices defining the range of sentences to process
def local_tf_idf(sentences, start, end):
    
    # Slice the input tokenized sentences
    sentences = sentences[start:end]

    # Create a set for storing unique words (vocabulary)
    local_vocabulary = set()

    # Collect unique words from the tokenized sentences
    for sent in sentences:
        local_vocabulary.update(sent)  # `sent` is already tokenized

    # Convert the vocabulary set into a list
    local_vocabulary_list = list(local_vocabulary)

    # Function to calculate TF (Term Frequency) for a single document (sentence)
    def calculate_tf(doc):
        """
        Compute term frequency for a document.
        Uses scaled frequency formula: TF = 0.5 + 0.5 * (freq / max_freq).
        """
        doc_words_freq = Counter(doc)  # Count the frequency of each term in the document
        max_frequency = max(doc_words_freq.values())  # Find the maximum frequency of any term in the document
        tf_of_words = {}
        for word, frequency in doc_words_freq.items():
            scaled_frequency = 0.5 + 0.5 * (frequency / max_frequency)  # Scale the frequency
            tf_of_words[word] = scaled_frequency
        return tf_of_words

    # Function to calculate IDF (Inverse Document Frequency) for all documents
    def calculate_idf(docs):
        """
        Compute inverse document frequency for a collection of documents.
        Uses smoothing: IDF = log(total_docs / (1 + freq)).
        """
        total_docs = len(docs)  # Total number of documents
        term_doc_freq = {}
        for doc in docs:
            unique_terms = set(doc)  # Get unique terms from each document
            for term in unique_terms:
                term_doc_freq[term] = term_doc_freq.get(term, 0) + 1  # Increment document frequency for the term
        return {term: math.log(total_docs / (1 + freq)) for term, freq in term_doc_freq.items()}  # Add smoothing

    # Compute TF for all sentences
    tf = [calculate_tf(sentence) for sentence in sentences]

    # Compute IDF for the local range of sentences
    idf = calculate_idf(sentences)

    # Compute TF-IDF scores
    tfidf = []
    for doc_tf in tf:
        tfidf.append({term: doc_tf.get(term, 0) * idf.get(term, 0) for term in doc_tf})  # TF-IDF calculation

    vector_contain_list = []  # List to store TF-IDF vectors for all sentences

    # Create vectors for each sentence
    for sentence_tfidf in tfidf:
        vector_of_list = {term: 0 for term in local_vocabulary_list}  # Initialize vector with zeros for all vocabulary terms
        for term, score in sentence_tfidf.items():
            vector_of_list[term] = score  # Update the vector with TF-IDF scores
        vector_contain_list.append(list(vector_of_list.values()))  # Append the vector to the list

    return vector_contain_list, local_vocabulary_list
