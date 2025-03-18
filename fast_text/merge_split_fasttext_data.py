import os
import random
import argparse

def split_dataset(folder_path, train_file, test_file, valid_file, num_train, num_test, num_valid):
    """
    Reads multiple text files from a folder, shuffles their lines, 
    and splits them into train, test, and validation datasets.

    Parameters:
        folder_path (str): Path to the folder containing labeled text files.
        train_file (str): Path to save the training dataset.
        test_file (str): Path to save the testing dataset.
        valid_file (str): Path to save the validation dataset.
        num_train (int): Number of sentences for training per file.
        num_test (int): Number of sentences for testing per file.
        num_valid (int): Number of sentences for validation per file.
    """

    # Open output files in write mode
    with open(train_file, "w", encoding="utf-8") as train_f, \
         open(test_file, "w", encoding="utf-8") as test_f, \
         open(valid_file, "w", encoding="utf-8") as valid_f:

        # Process each text file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):  # Ensure it's a text file
                file_path = os.path.join(folder_path, filename)
                
                with open(file_path, "r", encoding="utf-8") as file:
                    lines = file.readlines()
                    
                    # Check if the file has enough data for splitting
                    if len(lines) < num_train + num_test + num_valid:
                        print(f" Skipping {filename}, not enough sentences.")
                        continue
                    
                    # Shuffle the sentences to ensure randomness
                    random.shuffle(lines)

                    # Split data into train, test, and validation sets
                    train_sentences = lines[:num_train]
                    test_sentences = lines[num_train:num_train + num_test]
                    valid_sentences = lines[num_train + num_test:num_train + num_test + num_valid]

                    # Append to respective files
                    train_f.writelines(train_sentences)
                    test_f.writelines(test_sentences)
                    valid_f.writelines(valid_sentences)

    print(f"Files created successfully:\n  - {train_file}\n  - {test_file}\n  - {valid_file}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Split labeled text files into training, testing, and validation sets.")

    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing text files.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to save the training dataset.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to save the testing dataset.")
    parser.add_argument("--valid_file", type=str, required=True, help="Path to save the validation dataset.")
    parser.add_argument("--num_train", type=int, default=9000, help="Number of training samples per file (default: 9000).")
    parser.add_argument("--num_test", type=int, default=500, help="Number of test samples per file (default: 500).")
    parser.add_argument("--num_valid", type=int, default=500, help="Number of validation samples per file (default: 500).")

    args = parser.parse_args()

    # Call the function with provided arguments
    split_dataset(args.folder_path, args.train_file, args.test_file, args.valid_file, args.num_train, args.num_test, args.num_valid)
