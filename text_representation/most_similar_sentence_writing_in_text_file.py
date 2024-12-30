import math
from collections import Counter
import re
from nltk.tokenize import word_tokenize
import numpy as np

# Preprocess the data (Remove punctuation and convert to lowercase)
def pre_process(text):
    text = re.sub(r'[\n\t]', ' ', text)             # Remove newlines and tabs
    text = re.sub(r'[^\w\s]', '', text.lower())     # Remove punctuation and convert to lowercase
    return  word_tokenize(text) or []               # Tokenize text into words

# File handling or input data(contain sentences)
with open("C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\sent_tokenisation_with_language_code\\english.txt", "r", encoding="utf-8") as f:
    # Creating a set "vocabulary" that contains all the unique words
    vocabulary = set()
    lines = f.readlines()  # Read all lines into memory at once

    # Pre-process lines and tokenize into words
    cleaned_sentences = [pre_process(line.strip()) for line in lines]
    
    #removing empty sentences
    cleaned_sentences = [sentence for sentence in cleaned_sentences if sentence]

    # Collect words for the vocabulary
    for sentence in cleaned_sentences:
        vocabulary.update(sentence)  # Add words to the vocabulary set

    vocabulary_list = list(vocabulary)  # Convert vocabulary set into a list for indexing
    
     
    # Calculating TF (Term Frequency)
    def calculate_tf(doc):
        term_freq = Counter(doc)  # Count frequency of each term in the document
        max_frequency = max(term_freq.values())  # Maximum term frequency
        tf = {}
        for term, count in term_freq.items():
            # Scaled frequency (log(1 + count) or scaled version of the frequency)
            scaled_frequency = 0.5 + 0.5 * (count / max_frequency)
            tf[term] = scaled_frequency
        return tf

    # Calculating IDF (Inverse Document Frequency)
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

    # Calculating TF-IDF (Term Frequency - Inverse Document Frequency)
    def calculate_tfidf(docs):
        tf = [calculate_tf(doc) for doc in docs]      # Calculate TF for each document
        idf = calculate_idf(docs)                     # Calculate IDF across all documents
        tfidf = []
        for doc_tf in tf:
            doc_tfidf = {term: doc_tf.get(term, 0) * idf.get(term, 0) for term in doc_tf}  # Calculate TF-IDF
            
            
            tfidf.append(doc_tfidf)
        return tfidf

# Compute TF-IDF
tfidf = calculate_tfidf(cleaned_sentences)


# Print the cleaned data
print("Cleaned Data (List of tokenized sentences):")
# for sentence in cleaned_sentences:
#     print(sentence)



vector_contain_list = []

# Iterate through each sentence's TF-IDF
for sentence_tfidf in tfidf:
    # Initialize vector with 0 for all vocabulary terms
    vector_of_list = {term: 0 for term in vocabulary_list}
    
    # Update vector with TF-IDF scores for terms in the sentence
    for term, score in sentence_tfidf.items():
        vector_of_list[term] = score
    
    # Append the complete vector for the sentence to the list
    vector_contain_list.append(list(vector_of_list.values()))
# print(vector_contain_list)
print(len(vector_contain_list))
    
    


#sentence similarity 

for first_sent in range(len(vector_contain_list)):
    cosine_similarity = 0
    matched_vector = -1  # Reset matched vector for each sentence

    for next_sent in range(len(vector_contain_list)):
        if next_sent == first_sent:  # Skip comparing the sentence with itself
            continue
        
        # Convert vectors to numpy arrays
        A = np.array(vector_contain_list[first_sent])
        B = np.array(vector_contain_list[next_sent])
        
        # Calculate dot product and magnitudes
        dot_product = np.dot(A, B)
        magnitude_A = np.linalg.norm(A)
        magnitude_B = np.linalg.norm(B)
        
        # Avoid division by zero
        if magnitude_A == 0 or magnitude_B == 0:
            similarity = 0
        else:
            similarity = dot_product / (magnitude_A * magnitude_B)
        
        # Update the most similar sentence
        if similarity > cosine_similarity:
            matched_vector = next_sent
            cosine_similarity = similarity
    
    # Output result for the current sentence
    #writing a sentence and most similar sentence in a 'sentence_similarity.txt' file 
    #here i use path of that file named with 'sentence_similarity.txt'
    
    with open("C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\model_making\\text_representation\\sentence_similarity.txt",'a',encoding='utf-8') as file :
        
        # Write results for the current sentence in the file 
        file.write(f"Sentence: {' '.join(cleaned_sentences[first_sent])}\n")

        

        if matched_vector == -1:
                file.write(f"No similar sentence found for rank {matched_vector+ 1}.\n")
        else:
                # Write the k-ranked similar sentence
            file.write(f"\n most Similar Sentence : {' '.join(cleaned_sentences[matched_vector])}\n")
            file.write(f"Cosine Similarity: {cosine_similarity:.4f}\n")

        file.write("\n")
