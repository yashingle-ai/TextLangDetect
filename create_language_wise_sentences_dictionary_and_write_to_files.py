"""Create language-wise sentences dictionary and write to files."""
import os
import argparse


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def write_lines_to_file(output_file, lines):
    """Write a list to a file."""
    with open(output_file, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines) + '\n')


def combine_text_label_lists_from_files_and_create_language_wise_sentence_dictionary(input_folder):
    """Combine text and label lists from files and create a language dictionary."""
    language_dict = {}
    if not os.path.isdir(input_folder):
        raise ValueError(f"The provided path '{input_folder}' is not a directory.")
    type_of_files = ['train', 'test', 'dev']
    for type_of_file in type_of_files:
        text_file = os.path.join(input_folder, f'{type_of_file}_sentences.txt') # Assuming the text files are named accordingly
        label_file = os.path.join(input_folder, f'{type_of_file}_labels.txt')
        if os.path.isfile(text_file) and os.path.isfile(label_file):
            with open(text_file, 'r', encoding='utf-8') as f_text, open(label_file, 'r', encoding='utf-8') as f_label:
                text_lines = f_text.readlines()
                label_lines = f_label.readlines()
                if len(text_lines) != len(label_lines):
                    raise ValueError(f"Text and label files for '{type_of_file}' do not match in length.")
                for text_line, label_line in zip(text_lines, label_lines):
                    language = label_line.strip()
                    sentence = text_line.strip()
                    if language not in language_dict:
                        language_dict[language] = []
                    language_dict[language].append(sentence)
        else:
            raise FileNotFoundError(f"Required files for '{type_of_file}' not found in '{input_folder}'.")
    return language_dict


def write_language_wise_sentences_to_files(language_dict, output_folder):
    """Write language-wise sentences to files."""
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    for language, sentences in language_dict.items():
        print(f"Writing sentences for language: {language}")
        print(f"Number of sentences: {len(sentences)}", f"for {language} language.")
        output_file = os.path.join(output_folder, f'{language}_sentences.txt')
        write_lines_to_file(output_file, sentences)
    print("Total sentences written to files:", sum(len(sentences) for sentences in language_dict.values()), f'across {len(language_dict)} languages.')


def main():
    """Main function to handle command line arguments and process files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', dest='inp', help="Enter the input folder path containing text and label files")
    parser.add_argument(
        '--output', dest='out', help="Enter the output folder path to save language-wise sentences")
    args = parser.parse_args()

    if not os.path.isdir(args.inp):
        raise ValueError(f"The provided input path '{args.inp}' is not a directory.")
    
    language_dict = combine_text_label_lists_from_files_and_create_language_wise_sentence_dictionary(args.inp)
    write_language_wise_sentences_to_files(language_dict, args.out)


if __name__ == '__main__':
    main()
