# Local TF-IDF: Considers each sentence (or document) independently for TF-IDF calculations. This means:
# The "corpus" for a given sentence is only that single sentence.
# The vocabulary and IDF values are derived from the sentence itself.

from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import numpy as np 
import pandas as pd 
from nltk.tokenize import word_tokenize
import re 


def pre_process(text):
    text = re.sub(r'[\n\t]', ' ', text)             # Remove newlines and tabs
    text = re.sub(r'[^\w\s]', '', text.lower())     # Remove punctuation and convert to lowercase
    tokens = word_tokenize(text)                   # Tokenize text into words
    return ' '.join(tokens)                        # Return as a single string for TfidfVectorizer



def main():
    parser = argparse.ArgumentParser(description="Find k most similar sentences using Euclidean distance.")
    parser.add_argument("input_file", type=str, help="Path to the input text file")
    parser.add_argument("output_file", type=str, help="Path to the output text file")
    parser.add_argument('k',type=int,help="enter the no of similar sentence that you want ")
    args = parser.parse_args()
    
    with open(args.input_file,'r',encoding='utf-8') as file:
        
        lines=file.readlines() 
        
        #preprocess :removing punctuation and removing extra whitespace and lowercase the words 
        cleaned_sentences = [pre_process(line.split('\t')[0]) for line in lines if line.strip()]
        
        #initialise the tf-idf vector 
        vectorizer=TfidfVectorizer()
        
        # Process each sentence independently for local TF-IDF
        local_tfidf_representations = []
        
        #loop for each sentence
        for sentence in cleaned_sentences:
            
            # process each sentence individual
            tf_idf_matrix=vectorizer.fit_transform([sentence])
            
            # individual vocabulary for each sentence 
            vocabulary=vectorizer.get_feature_names_out()
            
            #here tf_idf_matrix for endividual 2d array with only one row like [[0.4 ,0.6,0.8,0.9]]
            # here we convert into array for further cosine similarity calculation 
            dence_matrix=tf_idf_matrix.toarray()[0]
            
            # now append sentence ,vocabulary of each sentence (individual) and dence matrix as single element in local tf_if_matrix for further calculation and comparision 
            local_tfidf_representations.append((sentence,vocabulary,dence_matrix))
            
        for idx, (sentence, feature_names, tfidf_vector) in enumerate(local_tfidf_representations):
            print(f"Sentence {idx + 1}: {sentence}")
            print("Features (Vocabulary):", feature_names)
            print("TF-IDF Vector:", tfidf_vector)
            print()
            
            
if __name__ == "__main__":
    main()

        

