# here we calculate TF-IDF for a specific range of sentences called local TF-IDF

import math
from collections import Counter  # Fixed import
import re
from nltk.tokenize import word_tokenize
import numpy as np



def pre_process(text):
    text = re.sub(r'[\n\t]', ' ', text)             # Remove newlines and tabs
    text = re.sub(r'[^\w\s]', '', text.lower())     # Remove punctuation and convert to lowercase
    return  word_tokenize(text) or []               # Tokenize text into words


#here we defined a function for calculation of tf idf for local sentences means some limitation for taking input for sentence 
# here we take input in function (sentences):that contain all sentences of file and start and end for slicing the sentece list 
 
def local_tf_idf(sentences, start, end):
    # Slice the input tokenized sentences
    sentences = sentences[start:end]
    
    # Create a set for storing unique words
    local_vocabulary = set()
    
    # Collect unique words from the tokenized sentences
    for sent in sentences:
        local_vocabulary.update(sent)  # sent is already tokenized
    
    local_vocabulary_list = list(local_vocabulary)  # Convert vocabulary set into list
    
    # Calculate TF for a single document (sentence)
    def calculate_tf(doc):
        doc_words_freq = Counter(doc)
        max_frequency = max(doc_words_freq.values())
        tf_of_words = {}
        for word, frequency in doc_words_freq.items():
            scaled_frequency = 0.5 + 0.5 * (frequency / max_frequency)
            tf_of_words[word] = scaled_frequency
        return tf_of_words
    
    # Calculate IDF for all documents
    def calculate_idf(docs):
        total_docs = len(docs)
        term_doc_freq = {}
        for doc in docs:
            unique_terms = set(doc)
            for term in unique_terms:
                term_doc_freq[term] = term_doc_freq.get(term, 0) + 1
        return {term: math.log(total_docs / (1 + freq)) for term, freq in term_doc_freq.items()}
    
    # Compute TF and IDF
    tf = [calculate_tf(sentence) for sentence in sentences]
    idf = calculate_idf(sentences)
    
    # Compute TF-IDF
    tfidf = []
    for doc_tf in tf:
        tfidf.append({term: doc_tf.get(term, 0) * idf.get(term, 0) for term in doc_tf})
    
    vector_contain_list = []

    # Create vectors for each sentence
    for sentence_tfidf in tfidf:
        vector_of_list = {term: 0 for term in local_vocabulary_list}
        for term, score in sentence_tfidf.items():
            vector_of_list[term] = score
        vector_contain_list.append(list(vector_of_list.values()))
    
    return vector_contain_list, local_vocabulary_list
