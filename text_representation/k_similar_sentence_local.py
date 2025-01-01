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
    return word_tokenize(text) or []               # Tokenize text into words


# Calculating TF (Term Frequency)
def calculate_tf(doc):
    doc_words_freq = Counter(doc)  # Count frequency of each term in the document
    max_frequency = max(doc_words_freq.values())  # Maximum term frequency
    tf_of_words = {word: 0.5 + 0.5 * (frequency / max_frequency)
                   for word, frequency in doc_words_freq.items()}  # Scaled frequency
    return tf_of_words


# Calculating IDF (Inverse Document Frequency)
def calculate_idf(docs):
    total_docs = len(docs)  # Number of documents (sentences)
    term_doc_freq = {}
    # Iterate through each document (sentence)
    for doc in docs:
        unique_terms = set(doc)  # Get unique terms from the document
        for term in unique_terms:
            term_doc_freq[term] = term_doc_freq.get(term, 0) + 1  # Increment document frequency for each term
    return {term: math.log(total_docs / (1 + freq)) for term, freq in term_doc_freq.items()}  # Add smoothing (1+freq)


# Compute TF-IDF vectors for all sentences
def compute_tfidf_vectors(sentences):
    # Create vocabulary from all sentences
    vocabulary = set()
    for sent in sentences:
        vocabulary.update(sent)  # Add words to the vocabulary set

    vocabulary_list = list(vocabulary)  # Convert vocabulary set into a list for indexing
    tf = [calculate_tf(sentence) for sentence in sentences]  # Calculate TF for each sentence
    idf = calculate_idf(sentences)  # Calculate IDF across all sentences

    # Calculate TF-IDF for each sentence
    tfidf = []
    for doc_tf in tf:
        tfidf.append({term: doc_tf.get(term, 0) * idf.get(term, 0) for term in doc_tf})  # TF-IDF calculation

    # Convert TF-IDF to vectors
    vectors = []
    for sentence_tfidf in tfidf:
        vector = {term: 0 for term in vocabulary_list}  # Initialize vector with zeros for all vocabulary terms
        for term, score in sentence_tfidf.items():
            vector[term] = score  # Update vector with TF-IDF scores for terms in the sentence
        vectors.append(list(vector.values()))  # Append vector to the list of vectors

    return vectors, vocabulary_list


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compute TF-IDF and find k similar sentences.")
    parser.add_argument("input_file", help="Path to the input text file.")
    parser.add_argument("output_file", help="Path to the output text file.")
    parser.add_argument("start", type=int, help="Start index of the range (0-based).")
    parser.add_argument("end", type=int, help="End index of the range (exclusive, 0-based).")
    parser.add_argument("-k", type=int, default=3, help="Number of similar sentences to find (default: 3).")
    args = parser.parse_args()

    # File handling and preprocessing
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()  # Read all lines into memory at once
        cleaned_sentences = [pre_process(line.strip()) for line in lines if line.strip()]  # Preprocess sentences

    # Ensure valid range
    if args.start < 0 or args.end > len(cleaned_sentences) or args.start >= args.end:
        raise ValueError("Invalid range. Ensure 0 <= start < end <= number of sentences.")

    # Slice the sentences for local TF-IDF calculation
    local_sentences = cleaned_sentences[args.start:args.end]  # Extract sentences within the specified range
    tfidf_vectors, vocabulary_list = compute_tfidf_vectors(local_sentences)  # Compute TF-IDF vectors

    # Find k most similar sentences for each sentence in the local range
    with open(args.output_file, "w", encoding="utf-8") as output_file:
        for first_sent_idx, vec1 in enumerate(tfidf_vectors):
            vec1_np = np.array(vec1)  # Convert vector to numpy array
            similarity_scores = []

            # Calculate similarity with all other local sentences
            for second_sent_idx, vec2 in enumerate(tfidf_vectors):
                if first_sent_idx == second_sent_idx:
                    continue  # Skip the same sentence

                vec2_np = np.array(vec2)  # Convert vector to numpy array
                dot_product = np.dot(vec1_np, vec2_np)  # Calculate dot product
                magnitude_1 = np.linalg.norm(vec1_np)  # Magnitude of the first vector
                magnitude_2 = np.linalg.norm(vec2_np)  # Magnitude of the second vector
                similarity = dot_product / (magnitude_1 * magnitude_2) if magnitude_1 and magnitude_2 else 0  # Cosine similarity
                similarity_scores.append((second_sent_idx, similarity))  # Append similarity score and index

            # Sort by similarity and get top k matches
            similarity_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity in descending order
            top_k_matches = similarity_scores[:args.k]  # Get top-k matches

            # Write results to the output file
            output_file.write(f"Sentence {args.start + first_sent_idx + 1}: {' '.join(local_sentences[first_sent_idx])}\n")
            for rank, (matched_idx, similarity) in enumerate(top_k_matches, start=1):
                output_file.write(f"  {rank}-Ranked Match: {' '.join(local_sentences[matched_idx])}\n")
                output_file.write(f"    Cosine Similarity: {similarity:.4f}\n")
            output_file.write("\n")


if __name__ == "__main__":
    main()
