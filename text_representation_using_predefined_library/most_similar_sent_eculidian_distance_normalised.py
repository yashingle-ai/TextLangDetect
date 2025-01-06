import re
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.tokenize import word_tokenize

# Preprocess function: remove newlines, tabs, and punctuation
def pre_process(text):
    text = re.sub(r'[\n\t]', ' ', text)  # Remove newlines and tabs
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and convert to lowercase
    tokens = word_tokenize(text)  # Tokenize text into words
    return ' '.join(tokens)  # Return as a single string for TfidfVectorizer

# Setup argparse to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate normalized Euclidean distances between sentences.")
    parser.add_argument("input_file", type=str, help="Path to the input text file.")
    parser.add_argument("output_file", type=str, help="Path to the output text file.")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        # Preprocess: remove punctuation, extra whitespace, and lowercase the words 
        cleaned_sentences = [pre_process(line.split('\t')[0]) for line in lines if line.strip()]

        # Initialize the TF-IDF vectorizer
        vectorizer = TfidfVectorizer()

        # Fit and transform sentences into TF-IDF vectors
        TF_idf_matrics = vectorizer.fit_transform(cleaned_sentences)

        # Convert sparse matrix to dense array
        dense_matrix = TF_idf_matrics.toarray()

        distance_of_sentence = []
        most_similar_sentence_index = []

        # Initialize max_distance
        max_distance = 0

        # Loop through all sentences to calculate distances and find max distance
        for row in dense_matrix:
            distance_of_sent = []

            for compare_row in dense_matrix:
                if np.array_equal(compare_row, row):
                    distance_of_sent.append(np.inf)  # Use np.inf to avoid self comparison
                else:
                    distance = np.linalg.norm(row - compare_row)
                    distance_of_sent.append(distance)

                    # Update max_distance if a larger distance is found
                    if distance > max_distance:
                        max_distance = distance

            # Get the indices of sorted distances
            distance_of_sent_indices = np.argsort(distance_of_sent)

            # Get the index of the second minimum (second-most similar sentence)
            second_min_index = distance_of_sent_indices[1]
            most_similar_sentence_index.append(second_min_index)

            # Normalize the distances using the maximum distance
            distance_of_sent_arr = np.array(distance_of_sent)
            normalized_dis = distance_of_sent_arr / max_distance  # Normalize with max distance

            distance_of_sentence.append(normalized_dis)

        # Writing the results to the output file
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for i, sentence in enumerate(cleaned_sentences):
                f.write(f"Sentence: {sentence}\n")
                z = most_similar_sentence_index[i]
                f.write(f"Most similar sentence: {cleaned_sentences[z]}\n")
                f.write(f"Normalized Euclidean distance: {distance_of_sentence[i][most_similar_sentence_index[i]]:.4f}\n")
                f.write("\n")

if __name__ == "__main__":
    main()
