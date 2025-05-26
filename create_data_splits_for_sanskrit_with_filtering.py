"""Create data splits for Sanskrit by merging all the sentences and filtering the data."""
from argparse import ArgumentParser
from glob import glob
import re
from string import punctuation
from random import sample
from random import shuffle


sanskrit_punctuation = punctuation + '।'


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def write_lines_to_file(lines, file_path):
    """Write lines to a file."""
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines) + '\n')


def filter_lines(lines):
    """Filter out lines from all the lines."""
    selected_lines = []
    for line in lines:
        if len(line.split()) <= 7:
            continue
        else:
            punc_split = re.split('[' + sanskrit_punctuation + ' ]', line)
            punc_split = [token.strip() for token in punc_split if token.strip()]
            if len(punc_split) <= 7:
                continue
            else:
                selected_lines.append(line)
    if len(selected_lines) >= 100:
        return sample(selected_lines, 100)
    else:
        return selected_lines


def read_files_and_filter_out_sentences(folder_path):
    """Read files and filter out sentences."""
    collected_sentences = []
    for file in glob(folder_path + '/*.txt'):
        lines = read_lines_from_file(file)
        filtered_sentences = filter_lines(lines)
        collected_sentences += filtered_sentences
    return collected_sentences


def add_labels_to_sentences(sentences):
    """Add language labels to sentences."""
    return [sentence + '\tsan' for sentence in sentences]


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser()
    parser.add_argument('--input', dest='inp', help='Enter the input folder path.')
    parser.add_argument('--train', dest='train', help='Enter the train file path.')
    parser.add_argument('--dev', dest='dev', help='Enter the dev file path.')
    parser.add_argument('--test', dest='test', help='Enter the test file path.')
    args = parser.parse_args()
    collected_sentences = read_files_and_filter_out_sentences(args.inp)
    print(len(collected_sentences))
    random_selected = sample(collected_sentences, 10000)
    shuffle(random_selected)
    train_sentences = random_selected[: 8000]
    dev_sentences = random_selected[8000: 9000]
    test_sentences = random_selected[9000:]
    train_sentences_with_labels = add_labels_to_sentences(train_sentences)
    dev_sentences_with_labels = add_labels_to_sentences(dev_sentences)
    test_sentences_with_labels = add_labels_to_sentences(test_sentences)
    write_lines_to_file(train_sentences_with_labels, args.train)
    write_lines_to_file(dev_sentences_with_labels, args.dev)
    write_lines_to_file(test_sentences_with_labels, args.test)


if __name__ == '__main__':
    main()
