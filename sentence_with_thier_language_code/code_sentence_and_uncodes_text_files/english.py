# take input file that contains sentence with thier uncode text file 
#output:- give a two outputfile that one consist a sentence and one contain unicode (in same index)

import argparse

def process_files(input_path, sentences_output_path, unicode_output_path):
    # Open input file for reading
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Open output files for writing sentences and unicode
    with open(sentences_output_path, 'w', encoding='utf-8') as sentences_file, \
         open(unicode_output_path, 'w', encoding='utf-8') as unicode_file:
        for line in lines:
            # Split the line on the tab character, handle lines without a tab
            sent_and_unicode = line.strip().split('\t')

            if len(sent_and_unicode) >= 2:  # Ensure there's both sentence and language code
                sentence = sent_and_unicode[0]
                unicode_lang = sent_and_unicode[1]
            else:
                sentence = sent_and_unicode[0]
                unicode_lang = "unknown"

            # Write the sentence to the sentences file
            sentences_file.write(f"{sentence}\n")

            # Write the Unicode to the unicode file
            unicode_file.write(f"{unicode_lang}\n")

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Process text file to separate sentences and unicode language codes.")
    
    # Add arguments for input and output file paths
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input file.")
    parser.add_argument('-s', '--sentences_output', type=str, required=True, help="Path to the output file for sentences.")
    parser.add_argument('-u', '--unicode_output', type=str, required=True, help="Path to the output file for unicode language codes.")
    
    # Parse the arguments
    args = parser.parse_args()

    # Process the files with the provided paths
    process_files(args.input, args.sentences_output, args.unicode_output)

if __name__ == "__main__":
    main()
