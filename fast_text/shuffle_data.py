import random
import argparse

def shuffle_file(input_file, output_file):
    """
    Reads a text file, shuffles its lines, and writes the shuffled content to an output file.

    Parameters:
        input_file (str): Path to the input file containing data.
        output_file (str): Path to save the shuffled data.
    """

    # Read the data
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Shuffle the data randomly
    random.shuffle(lines)

    # Write shuffled data to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f" Shuffled data saved to: {output_file}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Shuffle the lines of a text file and save the result.")
    
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the shuffled text file.")

    args = parser.parse_args()

    # Call function with provided arguments
    shuffle_file(args.input_file, args.output_file)
