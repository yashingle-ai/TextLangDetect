import math
from collections import Counter
import re
from nltk.tokenize import word_tokenize
import numpy as np


def pre_process(text):
    text = re.sub(r'[\n\t]', ' ', text)             # Remove newlines and tabs
    text = re.sub(r'[^\w\s]', '', text.lower())     # Remove punctuation and convert to lowercase
    return word_tokenize(text) or []


def calculate_tf(doc):
    doc_words_freq = Counter(doc)
    max_frequency = max(doc_words_freq.values())
    tf_of_words = {word: 0.5 + 0.5 * (frequency / max_frequency)
                   for word, frequency in doc_words_freq.items()}
    return tf_of_words


def calculate_idf(docs):
    total_docs = len(docs)
    term_doc_freq = {}
    for doc in docs:
        unique_terms = set(doc)
        for term in unique_terms:
            term_doc_freq[term] = term_doc_freq.get(term, 0) + 1
    return {term: math.log(total_docs / (1 + freq)) for term, freq in term_doc_freq.items()}


def compute_tfidf_vectors(sentences):
    vocabulary = set()
    for sent in sentences:
        vocabulary.update(sent)

    vocabulary_list = list(vocabulary)
    tf = [calculate_tf(sentence) for sentence in sentences]
    idf = calculate_idf(sentences)

    tfidf = []
    for doc_tf in tf:
        tfidf.append({term: doc_tf.get(term, 0) * idf.get(term, 0) for term in doc_tf})

    vectors = []
    for sentence_tfidf in tfidf:
        vector = {term: 0 for term in vocabulary_list}
        for term, score in sentence_tfidf.items():
            vector[term] = score
        vectors.append(list(vector.values()))

    return vectors, vocabulary_list


# File handling and preprocessing
with open("C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\sent_tokenisation_with_language_code\\english.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    cleaned_sentences = [pre_process(line.strip()) for line in lines if line.strip()]

# Get start and end indices for the local range
start = int(input("Enter the start index of the range: "))
end = int(input("Enter the end index of the range: "))

# Ensure valid range
if start < 0 or end > len(cleaned_sentences) or start >= end:
    raise ValueError("Invalid range. Ensure 0 <= start < end <= number of sentences.")

# Slice the sentences for local TF-IDF calculation
local_sentences = cleaned_sentences[start:end]
tfidf_vectors, vocabulary_list = compute_tfidf_vectors(local_sentences)

# Set the value of k
k = int(input("Enter the value of k: "))

# Find k most similar sentences for each sentence in the local range
with open("C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\model_making\\text_representation\\local_tf_idf_text.txt", "w", encoding="utf-8") as output_file:
    for first_sent_idx, vec1 in enumerate(tfidf_vectors):
        vec1_np = np.array(vec1)
        similarity_scores = []

        # Calculate similarity with all other local sentences
        for second_sent_idx, vec2 in enumerate(tfidf_vectors):
            if first_sent_idx == second_sent_idx:
                continue  # Skip the same sentence

            vec2_np = np.array(vec2)
            dot_product = np.dot(vec1_np, vec2_np)
            magnitude_1 = np.linalg.norm(vec1_np)
            magnitude_2 = np.linalg.norm(vec2_np)
            similarity = dot_product / (magnitude_1 * magnitude_2) if magnitude_1 and magnitude_2 else 0
            similarity_scores.append((second_sent_idx, similarity))

        # Sort by similarity and get top k matches
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_matches = similarity_scores[:k]

        # Write results to the output file
        output_file.write(f"Sentence {start + first_sent_idx + 1}: {' '.join(local_sentences[first_sent_idx])}\n")
        for rank, (matched_idx, similarity) in enumerate(top_k_matches, start=1):
            output_file.write(f"  {rank}-Ranked Match: {' '.join(local_sentences[matched_idx])}\n")
            output_file.write(f"    Cosine Similarity: {similarity:.4f}\n")
        output_file.write("\n")
