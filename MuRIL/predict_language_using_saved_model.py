"""Language Identification using saved fine-tuned model."""
from argparse import ArgumentParser
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import TextClassificationPipeline
from datasets import Dataset
import torch
import os
from pickle import load
from huggingface_hub import login
from huggingface_hub import create_repo
from huggingface_hub import HfApi


# this is an cased muril base model
tokenizer_model = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
device = torch.device('cuda:0')


def load_object_from_pickle(pickle_file):
    """Load the object from the pickle file."""
    with open(pickle_file, 'rb') as pickle_load:
        data_object = load(pickle_load)
        return data_object


def preprocess_function(examples):
    """Preprocess function for processing the data."""
    tokenized_inputs = tokenizer(examples['text'], truncation=True, max_length=256, padding='max_length')
    return tokenized_inputs


def read_lines_from_file(file_path):
    """
    Read lines from a file.

    Args:
    file_path: Enter the input file path.

    Returns: Lines read from the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def write_lines_to_file(lines, file_path):
    """
    Write lines to a file.

    Args:
    lines: Lines to be written to the file.
    file_path: Enter the output file path.

    Returns: None
    """
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines))


def main():
    """
    Pass arguments and call functions here.

    Args: None

    Returns: None
    """
    parser = ArgumentParser(description='This program is about finetuning a sentiment analyzer model.')
    parser.add_argument('--test', dest='te', help='Enter the test data in CSV format.')
    parser.add_argument('--model', dest='mod', help='Enter the model directory.')
    parser.add_argument('--output', dest='o', help='Enter the output file path for predictions.')
    args = parser.parse_args()
    test_dataset = read_lines_from_file(args.te)[1:]  # skip header
    test_texts = list(zip(*[line.split('\t') for line in test_dataset]))[0]
    test_dataset_raw = Dataset.from_dict({'text': test_texts})
    # load the index to label dictionary from pickle file
    index_to_label_dict = load_object_from_pickle('index-to-label-dict.pkl')
    num_labels = len(index_to_label_dict)
    print(f'Number of labels: {num_labels}')
    # create the tokenized dataset
    print(len(test_dataset))
    test_tokenized_dataset = test_dataset_raw.map(preprocess_function, batched=True)
    test_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    # Load the model from the specified directory
    print('Loading model from:', args.mod)
    model = AutoModelForSequenceClassification.from_pretrained(args.mod)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0)
    # Save the predictions to the output file
    predictions_test = pipe(test_tokenized_dataset['text'], truncation=True, max_length=256)
    actual_labels_test = []
    for prediction in predictions_test:
        pred_label = prediction['label']
        pred_index = pred_label.split('_')[1]
        actual_labels_test.append(index_to_label_dict[int(pred_index)])
    write_lines_to_file(actual_labels_test, args.o)


if __name__ == '__main__':
    main()
