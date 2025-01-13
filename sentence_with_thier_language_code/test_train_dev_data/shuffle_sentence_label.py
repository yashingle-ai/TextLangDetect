# here i take input of two file one is containing sentence one is conrtaining with thier labels and 
# now i shuffle the sentence and label with the help of this code 

import argparse
import random

def split_shutfle_file(sentence_file, label_file, shuffle_sentence_file, shuffle_label_file):
    # Read sentences from the input sentence file
    with open(sentence_file, 'r', encoding='utf-8') as sent_f:
        sentence = sent_f.readlines()

    # Read labels from the input label file
    with open(label_file, 'r', encoding='utf-8') as label_f:
        label = label_f.readlines()

    # Ensure the number of sentences matches the number of labels
    assert len(sentence) == len(label), "Mismatch between number of sentences and labels"

    # Combine sentences and labels into a single list of tuples
    sentence_with_label = list(zip(sentence, label))

    # Shuffle the combined list with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(sentence_with_label)

    # Separate the shuffled data back into sentences and labels
    _sentence, _label = zip(*sentence_with_label)

    # Write shuffled sentences to the output sentence file
    with open(shuffle_sentence_file, 'w', encoding='utf-8') as shuffle_sent_f:
        shuffle_sent_f.writelines(_sentence)

    # Write shuffled labels to the output label file
    with open(shuffle_label_file, 'w', encoding='utf-8') as shuffle_label_f:
        shuffle_label_f.writelines(_label)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Shuffle sentences and labels while maintaining their correspondence.")
    parser.add_argument('-s', '--sentence_file', required=True, help="Path to the input sentence file.")
    parser.add_argument('-l', '--label_file', required=True, help="Path to the input label file.")
    parser.add_argument('-o_s', '--output_sentence_file', required=True, help="Path to the output shuffled sentence file.")
    parser.add_argument('-o_l', '--output_label_file', required=True, help="Path to the output shuffled label file.")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with provided arguments
    split_shutfle_file(args.sentence_file, args.label_file, args.output_sentence_file, args.output_label_file)
