import math
from collections import Counter
import re
from nltk.tokenize import word_tokenize
import numpy as np
import argparse
import os

def pre_process(text):
    # """
    # Preprocess the data by removing punctuation and converting to lowercase.
    # """
    text = re.sub(r'[\n\t]', ' ', text)             # Remove newlines and tabs
    text = re.sub(r'[^\w\s]', '', text.lower())     # Remove punctuation and convert to lowercase
    return word_tokenize(text) or []                 # Tokenize text into words

def calculate_tf(doc):
    # """
    # Calculate Term Frequency (TF) for a document.
    # """
    term_freq = Counter(doc)
    max_frequency = max(term_freq.values())
    tf = {}
    for term, count in term_freq.items():
        scaled_frequency = 0.5 + 0.5 * (count / max_frequency)
        tf[term] = scaled_frequency
    return tf

def calculate_idf(docs):
    # """
    # Calculate Inverse Document Frequency (IDF) for a collection of documents.
    # """
    total_docs = len(docs)
    term_doc_freq = {}

    for doc in docs:
        unique_terms = set(doc)
        for term in unique_terms:
            term_doc_freq[term] = term_doc_freq.get(term, 0) + 1

    idf = {}
    for term, doc_freq in term_doc_freq.items():
        scaled_idf = math.log(total_docs / (1 + doc_freq))
        idf[term] = scaled_idf
    return idf

def calculate_tfidf(docs):
    # """
    # Calculate TF-IDF for a collection of documents.
    # """
    tf = [calculate_tf(doc) for doc in docs]
    idf = calculate_idf(docs)
    tfidf = []
    for doc_tf in tf:
        doc_tfidf = {term: doc_tf.get(term, 0) * idf.get(term, 0) for term in doc_tf}
        tfidf.append(doc_tfidf)
    return tfidf

def main():
    #here the input file is english.txt from 
    parser = argparse.ArgumentParser(description="TF-IDF and sentence similarity computation.")
    parser.add_argument("input_file", help="Path to the input text file containing sentences.")
    parser.add_argument("output_file", help="Path to the output file to save results.")
    parser.add_argument("-k", type=int, default=3, help="Number of similar sentences to find (default: 3).")
    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.input_file):
        print(f"Error: File '{args.input_file}' does not exist.")
        return

    # Read input file
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Preprocess sentences
    cleaned_sentences = [pre_process(line.strip()) for line in lines]
    cleaned_sentences = [sentence for sentence in cleaned_sentences if sentence]

    # Create vocabulary
    vocabulary = set()
    for sentence in cleaned_sentences:
        vocabulary.update(sentence)
    vocabulary_list = list(vocabulary)

    # Compute TF-IDF
    tfidf = calculate_tfidf(cleaned_sentences)

    # Generate vectors for each sentence
    vector_contain_list = []
    for sentence_tfidf in tfidf:
        vector_of_list = {term: 0 for term in vocabulary_list}
        for term, score in sentence_tfidf.items():
            vector_of_list[term] = score
        vector_contain_list.append(list(vector_of_list.values()))

    # Find k similar sentences
    with open(args.output_file, "w", encoding="utf-8") as file:
        for first_sent in range(len(vector_contain_list)):
            index_of_k_most_similar_sent = []
            cosine_similarity_of_k_most_similar_sent = []

            for _ in range(args.k):
                cosine_similarity = 0
                matched_vector = -1

                for next_sent in range(len(vector_contain_list)):
                    if next_sent in index_of_k_most_similar_sent or first_sent == next_sent:
                        continue

                    A = np.array(vector_contain_list[first_sent])
                    B = np.array(vector_contain_list[next_sent])

                    dot_product = np.dot(A, B)
                    magnitude_A = np.linalg.norm(A)
                    magnitude_B = np.linalg.norm(B)

                    similarity = dot_product / (magnitude_A * magnitude_B) if magnitude_A and magnitude_B else 0

                    if similarity > cosine_similarity:
                        matched_vector = next_sent
                        cosine_similarity = similarity

                index_of_k_most_similar_sent.append(matched_vector)
                cosine_similarity_of_k_most_similar_sent.append(cosine_similarity)

            file.write(f"Sentence: {' '.join(cleaned_sentences[first_sent])}\n")
            for i, index_of_sentence in enumerate(index_of_k_most_similar_sent):
                if index_of_sentence == -1:
                    file.write(f"No similar sentence found for rank {i + 1}.\n")
                else:
                    file.write(f"{i + 1}-Ranked Similar Sentence: {' '.join(cleaned_sentences[index_of_sentence])}\n")
                    file.write(f"Cosine Similarity: {cosine_similarity_of_k_most_similar_sent[i]:.4f}\n")
            file.write("\n")

if __name__ == "__main__":
    main()
