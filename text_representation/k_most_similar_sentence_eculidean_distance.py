import math
from collections import Counter
import re
from nltk.tokenize import word_tokenize
import numpy as np
import argparse

# Preprocess the data (Remove punctuation and convert to lowercase)
def pre_process(text):
    text = re.sub(r'[\n\t]', ' ', text)             # Remove newlines and tabs
    text = re.sub(r'[^\w\s]', '', text.lower())     # Remove punctuation and convert to lowercase
    return word_tokenize(text) or []                 # Tokenize text into words

def calculate_tf(doc):
    term_freq = Counter(doc)  # Count frequency of each term in the document
    max_frequency = max(term_freq.values())  # Maximum term frequency
    tf = {}
    for term, count in term_freq.items():
        # Scaled frequency (log(1 + count) or scaled version of the frequency)
        scaled_frequency = 0.5 + 0.5 * (count / max_frequency)
        tf[term] = scaled_frequency
    return tf

def calculate_idf(docs):
    total_docs = len(docs)  # Number of documents (sentences)
    term_doc_freq = {}

    # Iterate through each document (sentence)
    for doc in docs:
        unique_terms = set(doc)  # Get unique terms from the document
        for term in unique_terms:
            if term not in term_doc_freq:
                term_doc_freq[term] = 0
            term_doc_freq[term] += 1

    # Calculate IDF for each term
    idf = {}
    for term, doc_freq in term_doc_freq.items():
        scaled_idf = math.log(total_docs / (1 + doc_freq))  # Add smoothing (1+doc_freq)
        idf[term] = scaled_idf
    return idf

def calculate_tfidf(docs):
    tf = [calculate_tf(doc) for doc in docs]      # Calculate TF for each document
    idf = calculate_idf(docs)                     # Calculate IDF across all documents
    tfidf = []
    for doc_tf in tf:
        doc_tfidf = {term: doc_tf.get(term, 0) * idf.get(term, 0) for term in doc_tf}  # Calculate TF-IDF
        tfidf.append(doc_tfidf)
    return tfidf

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Find k most similar sentences using Euclidean distance.")
    parser.add_argument("sentence_with_thier_language_code/english.txt", type=str, help="Path to the input text file")
    parser.add_argument("text_representation/k_most_similar_sentence_eculidean_distance.py", type=str, help="Path to the output text file")
    parser.add_argument("k", type=int, help="Number of similar sentences to find")
    args = parser.parse_args()

    # Read input file
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()  # Read all lines into memory at once

    # Pre-process lines and tokenize into words
    cleaned_sentences = [pre_process(line.strip()) for line in lines]

    # Remove empty sentences
    cleaned_sentences = [sentence for sentence in cleaned_sentences if sentence]

    # Create vocabulary
    vocabulary = set(word for sentence in cleaned_sentences for word in sentence)
    vocabulary_list = list(vocabulary)

    # Calculate TF-IDF
    tfidf = calculate_tfidf(cleaned_sentences)

    # Create TF-IDF vectors
    vector_contain_list = []
    for sentence_tfidf in tfidf:
        vector_of_list = {term: 0 for term in vocabulary_list}
        for term, score in sentence_tfidf.items():
            vector_of_list[term] = score
        vector_contain_list.append(list(vector_of_list.values()))

    # Find k most similar sentences
    with open(args.output_file, "w", encoding="utf-8") as file:
        for first_sent in range(len(vector_contain_list)):
            index_of_k_most_similar_sent = []
            distance_of_k_most_similar_sent = []

            for loop in range(args.k):
                sent_distance = float('inf')
                matched_vector = -1

                for next_sent in range(len(vector_contain_list)):
                    if next_sent in index_of_k_most_similar_sent or first_sent == next_sent:
                        continue

                    A = np.array(vector_contain_list[first_sent])
                    B = np.array(vector_contain_list[next_sent])

                    distance_cur_sent = np.linalg.norm(A - B)

                    if distance_cur_sent < sent_distance:
                        matched_vector = next_sent
                        sent_distance = distance_cur_sent

                index_of_k_most_similar_sent.append(matched_vector)
                distance_of_k_most_similar_sent.append(sent_distance)

            # Write results to the output file
            file.write(f"Sentence: {' '.join(cleaned_sentences[first_sent])}\n")

            for i, index_of_sentence in enumerate(index_of_k_most_similar_sent):
                if index_of_sentence == -1:
                    file.write(f"No similar sentence found for rank {i + 1}.\n")
                else:
                    file.write(f"Rank {i + 1} Similar Sentence: {' '.join(cleaned_sentences[index_of_sentence])}\n")
                    file.write(f"Euclidean Distance: {distance_of_k_most_similar_sent[i]:.4f}\n")

            file.write("\n")

if __name__ == "__main__":
    main()
