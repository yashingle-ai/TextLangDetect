import argparse
import os
import random

# Function to merge labeled files and split into train, validation, and test sets
def process_labeled_data(labeled_folder, combined_file, train_file, valid_file, test_file):
    all_lines = []
    
    #  Merge all labeled files into one
    for filename in os.listdir(labeled_folder):
        file_path = os.path.join(labeled_folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            all_lines.extend(lines)
    
    print(f"Total sentences before shuffling: {len(all_lines)}")
    
    #  Shuffle the labeled data
    random.shuffle(all_lines)
    
    #  Split into Train (80%), Validation (10%), Test (10%)
    total_samples = len(all_lines)
    train_size = int(0.8 * total_samples)
    valid_size = int(0.1 * total_samples)
    
    train_data = all_lines[:train_size]
    valid_data = all_lines[train_size:train_size + valid_size]
    test_data = all_lines[train_size + valid_size:]
    
    #  Save files
    with open(combined_file, "w", encoding="utf-8") as f:
        f.writelines(all_lines)
    
    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(train_data)
    
    with open(valid_file, "w", encoding="utf-8") as f:
        f.writelines(valid_data)
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.writelines(test_data)
    
    print(f"Total Samples: {total_samples}")
    print(f"Train: {len(train_data)}, Validation: {len(valid_data)}, Test: {len(test_data)}")
    print(f"Files saved: {train_file}, {valid_file}, {test_file}")

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge labeled text data and split into train, validation, and test sets.")
    parser.add_argument("--labeled_folder", type=str, required=True, help="Path to the folder containing labeled text files.")
    parser.add_argument("--combined_file", type=str, required=True, help="Path to save the combined labeled text file.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to save the training data file.")
    parser.add_argument("--valid_file", type=str, required=True, help="Path to save the validation data file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to save the test data file.")
    
    args = parser.parse_args()
    
    process_labeled_data(args.labeled_folder, args.combined_file, args.train_file, args.valid_file, args.test_file)
