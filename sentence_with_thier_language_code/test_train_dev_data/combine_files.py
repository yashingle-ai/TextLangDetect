# code for combine files 

#import argparse
import argparse

# creating function for combining files 
def combine_files(input_files, output_file):
    combined_lines = []

    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                combined_lines.extend(file.readlines())
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    try:
        with open(output_file, 'w', encoding='utf-8') as output_f:
            output_f.writelines(combined_lines)
        print(f"Combined files written to {output_file}")
    except Exception as e:
        print(f"Error writing to output file {output_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine text files into a single file.")
    parser.add_argument(
        '-i', '--input', 
        nargs='+', 
        required=True, 
        help="List of input file paths to combine."
    )
    parser.add_argument(
        '-o', '--output', 
        required=True, 
        help="Output file path to save the combined content."
    )

    args = parser.parse_args()

    combine_files(args.input, args.output)

