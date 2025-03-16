import argparse

# Function to add labels to each sentence in the input file
def label_data(input_file, output_file, language_label):
    # Open input file and create labeled output file
    with open(input_file, "r", encoding="utf-8") as in_f, open(output_file, "w", encoding="utf-8") as out_f:
        for line in in_f:
            line = line.strip()
            if line:  # Avoid empty lines
                labeled_sentence = f"__label__{language_label} {line}\n"  # Add label prefix
                out_f.write(labeled_sentence)  # Write to output file
    
    print(f"Labeled file saved: {output_file}")

# Main execution block
if __name__ == "__main__":
    # Set up argument parser to take input file, output file, and language label from the command line
    parser = argparse.ArgumentParser(description="Label text data for FastText.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text file.")  # Input file argument
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the labeled output file.")  # Output file argument
    parser.add_argument("--language_label", type=str, required=True, help="Language label to prepend.")  # Language label argument
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Call the function to process the file
    label_data(args.input_file, args.output_file, args.language_label)
