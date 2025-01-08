#bag of words text representation  without using library 

import math
from collections import Counter
import re
from nltk.tokenize import word_tokenize
import argparse
import os


def pre_process(text):
    text = re.sub(r'[\n\t]', ' ', text)             # Remove newlines and tabs
    text = re.sub(r'[^\w\s]', '', text.lower())     # Remove punctuation and convert to lowercase
    return word_tokenize(text)   # Tokenize text into words

def calculate_tf(doc):
    if not doc:  # Handle empty documents
        return {}
    term_freq = Counter(doc)
    max_frequency = max(term_freq.values())
    tf = {term: 0.5 + 0.5 * (count / max_frequency) for term, count in term_freq.items()}
    return tf

def calculate_idf(docs):
    total_docs = len(docs)
    term_doc_freq = {}
    for doc in docs:
        unique_terms = set(doc)
        for term in unique_terms:
            term_doc_freq[term] = term_doc_freq.get(term, 0) + 1
    idf = {term: math.log(total_docs / (1 + doc_freq)) for term, doc_freq in term_doc_freq.items()}
    return idf

def calculate_tfidf(docs):
    tf = [calculate_tf(doc) for doc in docs]
    idf = calculate_idf(docs)
    tfidf = [{term: doc_tf.get(term, 0) * idf.get(term, 0) for term in doc_tf} for doc_tf in tf]
    return tfidf

def main():
    parser=argparse.ArgumentParser(description="with the hepl of tf-idf compute bag_of_words")
    parser.add_argument("input_file",type=str,help="path to input text file ")
    args=parser.parse_args() 
    
    
    # Ensure input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: File '{args.input_file}' does not exist.")
        return
   
# Read input file
    with open(args.input_file, "r", encoding="utf-8") as f:
        
        #creating set for storing words(unique) 
        vocabulary = set()
        
        lines = f.readlines()
        
        #preprocess :removing punctuation and removing extra whitespace and lowercase the words 
        cleaned_sentences = [pre_process(line.strip()) for line in lines if line.strip()]
        cleaned_sentences = [sentence for sentence in cleaned_sentences if sentence]
        
        #update vocabulary set wtih the words of sentence 
        for sentence in cleaned_sentences:
            vocabulary.update(sentence)
        vocabulary_list = list(vocabulary)


    # calculating tf-idf for sentences that complete the preprocess
    tfidf = calculate_tfidf(cleaned_sentences)
    
    #create vector for represent sentence with tf-idf values of words according to vocabulary list 
    vector_contain_list = []
    
    #if word present in the given sentence then change the value with zero of vector contain list if it is not present the its zero
    for sentence_tfidf in tfidf:
        vector_of_list = {term: 0 for term in vocabulary_list}
        for term, score in sentence_tfidf.items():
            vector_of_list[term] = score
        vector_contain_list.append(list(vector_of_list.values()))

    print("Final Vectors:", vector_contain_list)

if __name__ == "__main__":
    main()

