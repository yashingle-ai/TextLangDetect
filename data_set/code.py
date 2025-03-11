import pyarrow.parquet as pq
from datasets import Dataset

# Define file paths
file_path = r"C:\\Medicine Recommendation System\\Data\\train-00100-of-00213.parquet"
output_file = r"C:\\Users\\yashi\\OneDrive\\Desktop\\dataset\\'tam_Taml'.txt"
target_language = 'tam_Taml'

# Load dataset
dset = Dataset(pq.read_table(file_path, memory_map=True))

# Debug: Print available languages
available_languages = set(dset["target_language"])
print("Unique target languages in dataset:", available_languages)

# Ensure target_language exists in dataset
if target_language not in available_languages:
    print(f"Error: '{target_language}' not found in dataset!")
    exit()

# Extract unique sentences
extracted_sentences = set()
for i in range(len(dset)):
    if dset[i]["target_language"].strip().lower() == target_language.lower():
        extracted_sentences.add(dset[i]["target_text"].strip())  # Normalize sentence
        
        if len(extracted_sentences) >= 15000:
            break  # Stop once 15k unique sentences are collected

# Write to file
with open(output_file, "a", encoding="utf-8") as f:
    for sentence in extracted_sentences:
        f.write(sentence + "\n")

print(f"Successfully written {len(extracted_sentences)} unique sentences for {target_language} to {output_file}")
