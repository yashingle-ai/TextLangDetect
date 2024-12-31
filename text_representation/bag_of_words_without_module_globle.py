import math
from collections import Counter
import re
from nltk.tokenize import word_tokenize

def pre_process(text):
    text = re.sub(r'[\n\t]', ' ', text)             # Remove newlines and tabs
    text = re.sub(r'[^\w\s]', '', text.lower())     # Remove punctuation and convert to lowercase
    return word_tokenize(text)                      # Tokenize text into words

# Read input file
with open("C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\sent_tokenisation_with_language_code\\english.txt", "r", encoding="utf-8") as f:
    vocabulary = set()
    lines = f.readlines()
    cleaned_sentences = [pre_process(line.strip()) for line in lines if line.strip()]
    cleaned_sentences = [sentence for sentence in cleaned_sentences if sentence]

    for sentence in cleaned_sentences:
        vocabulary.update(sentence)
    vocabulary_list = list(vocabulary)

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

tfidf = calculate_tfidf(cleaned_sentences)
vector_contain_list = []

for sentence_tfidf in tfidf:
    vector_of_list = {term: 0 for term in vocabulary_list}
    for term, score in sentence_tfidf.items():
        vector_of_list[term] = score
    vector_contain_list.append(list(vector_of_list.values()))

print("Final Vectors:", vector_contain_list)
