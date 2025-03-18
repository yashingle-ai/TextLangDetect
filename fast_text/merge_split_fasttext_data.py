import os
import random

# Folder containing the 22 label_language_code.txt files
folder_path = "C:\\Users\\yashi\\OneDrive\\Desktop\\fasttext\\label_data_10k"  # Change this to your actual folder path

# Output file paths
train_file = "C:\\Users\\yashi\\OneDrive\\Desktop\\fasttext\\data_set_10k\\train.txt"
test_file = "C:\\Users\\yashi\\OneDrive\\Desktop\\fasttext\\data_set_10k\\test.txt"
valid_file = "C:\\Users\\yashi\\OneDrive\\Desktop\\fasttext\\data_set_10k\\valid.txt"

# Parameters
num_train = 9000
num_test = 500
num_valid = 500

# Open output files in append mode
with open(train_file, "w", encoding="utf-8") as train_f, \
     open(test_file, "w", encoding="utf-8") as test_f, \
     open(valid_file, "w", encoding="utf-8") as valid_f:

    # Process each file
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Ensure it's a text file
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                
                if len(lines) < num_train + num_test + num_valid:
                    print(f"Skipping {filename}, not enough sentences.")
                    continue
                
                # Shuffle sentences in each file
                random.shuffle(lines)

                # Split sentences
                train_sentences = lines[:num_train]
                test_sentences = lines[num_train:num_train + num_test]
                valid_sentences = lines[num_train + num_test:num_train + num_test + num_valid]

                # Append to respective files
                train_f.writelines(train_sentences)
                test_f.writelines(test_sentences)
                valid_f.writelines(valid_sentences)

print("Files created successfully: train.txt, test.txt, valid.txt")
