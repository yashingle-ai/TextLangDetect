import math
from collections import Counter
import re
from nltk.tokenize import word_tokenize
import numpy as np
import argparse

def pre_process(text):
    text = re.sub(r'[\n\t]', ' ', text)             # Remove newlines and tabs
    text = re.sub(r'[^\w\s]', '', text.lower())     # Remove punctuation and convert to lowercase
    return word_tokenize(text) or []                 # Tokenize text into words

# Function for calculating local TF-IDF
# Takes tokenized sentences and start/end indices as input
def local_tf_idf(sentences, start, end):
    # Slice the input tokenized sentences
    sentences = sentences[start:end]
    
    # Create a set for storing unique words
    local_vocabulary = set()
    
    # Collect unique words from the tokenized sentences
    for sent in sentences:
        local_vocabulary.update(sent)  # `sent` is already tokenized
    
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

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate local TF-IDF and cosine similarity for sentences.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text file.")
    parser.add_argument("--start", type=int, required=True, help="Start index for local TF-IDF computation.")
    parser.add_argument("--end", type=int, required=True, help="End index for local TF-IDF computation.")
    args = parser.parse_args()

    # File handling and preprocessing
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        cleaned_sentences = [pre_process(line.strip()) for line in lines]
        cleaned_sentences = [sentence for sentence in cleaned_sentences if sentence]  # Remove empty sentences

    # Define the local range for TF-IDF computation
    start = args.start
    end = args.end

    # Ensure the indices are valid
    if start < 0 or end > len(cleaned_sentences) or start >= end:
        raise ValueError("Invalid range. Ensure 0 <= start < end <= number of sentences.")

    local_tfidf_vectors, local_vocabulary = local_tf_idf(cleaned_sentences, start, end)

    # Print the local TF-IDF vectors
    print("Local TF-IDF Vectors:")
    for idx, vector in enumerate(local_tfidf_vectors):
        print(f"Sentence {start + idx}: {vector}")

    # Compute cosine similarity for local sentences
    for first_sent in range(len(local_tfidf_vectors)):
        cosine_similarity = 0
        matched_vector = -1
        for next_sent in range(len(local_tfidf_vectors)):
            if next_sent == first_sent:
                continue
            A = np.array(local_tfidf_vectors[first_sent])
            B = np.array(local_tfidf_vectors[next_sent])
            dot_product = np.dot(A, B)
            magnitude_A = np.linalg.norm(A)
            magnitude_B = np.linalg.norm(B)
            similarity = dot_product / (magnitude_A * magnitude_B) if magnitude_A and magnitude_B else 0
            if similarity > cosine_similarity:
                cosine_similarity = similarity
                matched_vector = next_sent
        print(f"Sentence {first_sent} is most similar to Sentence {matched_vector} with similarity {cosine_similarity:.4f}")
