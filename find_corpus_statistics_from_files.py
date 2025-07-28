"""Find different corpus statistics in text files.
This script reads text files, tokenizes the content, and writes the output to specified files.
It supports both file and folder inputs, processing each file individually."""
import os
import argparse
import pandas as pd


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def find_corpus_statistics_from_files(input_folder):
    """Find corpus statistics from files in the input folder."""
    if not os.path.isdir(input_folder):
        raise ValueError(f"The provided path '{input_folder}' is not a directory.")
    
    statistics = {}
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            language = file_name.replace('_sentences.txt', '')  # Assuming the language is part of the filename
            print(f"Processing file: {file_name} for language: {language}")
            # Read lines from the file
            file_path = os.path.join(input_folder, file_name)
            lines = read_lines_from_file(file_path)
            statistics[file_name] = {
                'lang': language,
                '#sents': len(lines),
                '#words': sum(len(line.split()) for line in lines),
                '#chars': sum(len(line) for line in lines),
                'avg_word_len': round(sum(len(word) for line in lines for word in line.split()) / max(1, sum(len(line.split()) for line in lines)), 3),
                'avg_sent_len': round(sum(len(line) for line in lines) / max(1, len(lines)), 3),
                '#unique_words': len(set(word for line in lines for word in line.split())),
                'TTR_words': round(len(set(word for line in lines for word in line.split())) / max(1, sum(len(line.split()) for line in lines)), 3)
            }
    return statistics


def write_statistics_to_csv_file(output_file, statistics):
    """Write statistics to a TSV file."""
    df = pd.DataFrame.from_dict(statistics, orient='index')
    df.to_csv(output_file, index=False)
    print(f"Statistics written to {output_file}")


def main():
    """Main function to handle command line arguments and process files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', dest='inp', help="Enter the input folder path containing text files")
    parser.add_argument(
        '--output', dest='out', help="Enter the output file path for statistics")
    args = parser.parse_args()

    if not args.inp or not args.out:
        raise ValueError("Both input folder and output file paths must be provided.")

    statistics = find_corpus_statistics_from_files(args.inp)
    write_statistics_to_csv_file(args.out, statistics)


if __name__ == "__main__":
    main()
