import re 
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import pandas as pd 
import numpy as np 
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity



def pre_process(text):
    text = re.sub(r'[\n\t]', ' ', text)             # Remove newlines and tabs
    text = re.sub(r'[^\w\s]', '', text.lower())     # Remove punctuation and convert to lowercase
    tokens = word_tokenize(text)                   # Tokenize text into words
    return ' '.join(tokens)                        # Return as a single string for TfidfVectorizer


def main():
    parser = argparse.ArgumentParser(description="Find k most similar sentences using Euclidean distance.")
    parser.add_argument("input_file", type=str, help="Path to the input text file")
    parser.add_argument("output_file", type=str, help="Path to the output text file")
    args = parser.parse_args()
    
    with open(args.input_file,'r',encoding='utf-8') as file:
        lines=file.readlines() 
        
        #preprocess :removing punctuation and removing extra whitespace and lowercase the words 
        cleaned_sentences = [pre_process(line.split('\t')[0]) for line in lines if line.strip()]
        
        #initialise the tf-idf vector 
        vectorizer=TfidfVectorizer()
        
        #with the help of fit method we first calculate tf_idf of each sentence and then transform the sentence into vector form according to vocabulary called feature
        #here fit create a vocabulary and calculate idf for each unique words and ready to tranform
        # For each document in the dataset, it calculates the TF (Term Frequency) for each term in the vocabulary.Combines the TF and IDF to calculate the final TF-IDF values.
        
        TF_idf_matrics=vectorizer.fit_transform(cleaned_sentences)
        
        #vocabulary with the help of getfeature_name_out method 
        vocabulary=vectorizer.get_feature_names_out()
        
        #converting into array form 
        dense_matrix=TF_idf_matrics.toarray()
        

        # Print the matrix with column headers
        print("Feature Names (Vocabulary):", vocabulary)
        print("\nBag of Words Representation (Matrix):\n")

        # Print each row (sentence) and its vector representation
        for i, row in enumerate(dense_matrix):
            print(f"Sentence {i + 1}: {row}")
        
        #calulating_similarity_score for each sentence with all other sentences 
            
        sentence_similarity_score=cosine_similarity(dense_matrix)
        
        #comparing the sentences
        
    with open(args.output_file,'w',encoding='utf-8')as f:
        for i,row in enumerate(sentence_similarity_score):
          
            #sorted the sentences comparsion of cosine similarity and return second most similar sentence score index 
            #here argsort in asceding order list of cosine similarity score for sentence with other sentence
            most_similar_idx = row.argsort()[-2]
            
            #second most similar sentence similarity score 
            similarity_score = row[most_similar_idx]
            
            #writing into file 
            f.write(f"Sentence {i + 1}: {cleaned_sentences[i]}\n"f"Most Similar to Sentence {most_similar_idx + 1}: {cleaned_sentences[most_similar_idx]}\n"f"Similarity Score: {similarity_score:.4f}\n\n")


if __name__ == "__main__":
    main()
