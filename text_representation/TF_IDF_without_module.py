import math
from collections import Counter
import re

def pre_process(text):
    # Remove punctuation and convert to lowercase
    # Do not remove punctuations ---
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()

def calculate_tf(doc):
    term_freq = Counter(doc)
    tf = {}                                                         # an empty dictionary to store term frequencies
    for term, count in term_freq.items():                           # Go through each word (term) and its frequency (count)
        scaled_frequency = math.log(1 + count)                      # Scale the frequency by taking log(1 + count)
        tf[term] = scaled_frequency                                 # Store the result in the dictionary

    return tf

def calculate_idf(docs):
    total_docs = len(docs)                             #currenly i am giving only one paragraph as input so there in total doc=1
    term_doc_freq = {}

    for doc in docs:
        unique_terms = set(doc)                       #now converting doc into set because we want to cal. uniquness of words  
        for term in unique_terms:
            if term not in term_doc_freq:
                term_doc_freq[term] = 0
            term_doc_freq[term] += 1

    
    idf = {}                                                                    # Create an empty dictionary to store IDF values
    for term, doc_freq in term_doc_freq.items():                                # Go through each term and the number of documents it appears in
    #    scaled_idf = math.log(1 + total_docs / doc_freq)                         # Calculate the scaled IDF for the term
       scaled_idf = math.log(total_docs / 1 + doc_freq)                         # Calculate the scaled IDF for the term, this is correct
       idf[term] = scaled_idf                                                   # Store the result in the dictionary

    return idf

def calculate_tfidf(docs):
    tf = [calculate_tf(doc) for doc in docs]
    idf = calculate_idf(docs)

    tfidf = []
    for doc_tf in tf:
        doc_tfidf = {term: doc_tf.get(term, 0) * idf.get(term, 0) for term in doc_tf}
        tfidf.append(doc_tfidf)

    return tfidf

# Example single paragraph
paragraph = """Education plays a vital role in our lives. It provides knowledge, skills, and the ability to make decisions. 
Every parent should ensure their children receive education. Education is the backbone of a nation's progress!"""

# Preprocess the text (remove punctuation and tokenize)
paragraph_tokens = pre_process(paragraph)

# Since there's only one document, wrap it in a list
docs = [paragraph_tokens]

# Compute TF-IDF
tfidf = calculate_tfidf(docs)

# Print the TF-IDF for the single paragraph
print("TF-IDF for the paragraph:")
for term, score in tfidf[0].items():
    print(f"  {term}: {score:.4f}")
