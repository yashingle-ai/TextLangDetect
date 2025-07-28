"""Language Identification using pretrained MuRIL model."""
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

# this is an cased muril base model
model_name = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device('cuda:0')


def load_object_from_pickle(pickle_file):
    """Load the object from the pickle file."""
    with open(pickle_file, 'rb') as pickle_load:
        data_object = load(pickle_load)
        return data_object


def preprocess_function(examples):
    """Preprocess function for processing the data."""
    tokenized_inputs = tokenizer(examples['text'], truncation=True, max_length=256, padding='max_length')
    tokenized_inputs['labels'] = [int(label) for label in examples['label']]
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
    parser.add_argument('--train', dest='tr', help='Enter the training data in CSV format.')
    parser.add_argument('--dev', dest='de', help='Enter the dev data in CSV format.')
    parser.add_argument('--test', dest='te', help='Enter the test data in CSV format.')
    parser.add_argument('--model', dest='mod', help='Enter the model directory.')
    parser.add_argument('--epoch', dest='ep', help='Enter the number of epochs.', type=int)
    parser.add_argument('--output_1', dest='o1', help='Enter the output file path for predictions.')
    parser.add_argument('--output_2', dest='o2', help='Enter the output file path for predictions.')
    args = parser.parse_args()
    train_dataset = read_lines_from_file(args.tr)[1:]  # skip header
    dev_dataset = read_lines_from_file(args.de)[1:]  # skip header
    test_dataset = read_lines_from_file(args.te)[1:]  # skip header
    train_texts, train_labels = list(zip(*[line.split('\t') for line in train_dataset]))
    train_dataset_raw = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    dev_texts, dev_labels = list(zip(*[line.split('\t') for line in dev_dataset]))
    dev_dataset_raw = Dataset.from_dict({'text': dev_texts, 'label': dev_labels})
    test_texts, test_labels = list(zip(*[line.split('\t') for line in test_dataset]))
    test_dataset_raw = Dataset.from_dict({'text': test_texts, 'label': test_labels})
    # load the index to label dictionary from pickle file
    index_to_label_dict = load_object_from_pickle('index-to-label-dict.pkl')
    num_labels = len(index_to_label_dict)
    print(f'Number of labels: {num_labels}')
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    # create the tokenized dataset
    print(len(train_dataset))
    print(len(test_dataset))
    print(len(dev_dataset))
    train_tokenized_dataset = train_dataset_raw.map(preprocess_function, batched=True)
    dev_tokenized_dataset = dev_dataset_raw.map(preprocess_function, batched=True)
    test_tokenized_dataset = test_dataset_raw.map(preprocess_function, batched=True)
    # print(train_tokenized_dataset.shape)
    train_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dev_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    training_args = TrainingArguments(
        output_dir=args.mod,
        overwrite_output_dir=True,
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=args.ep,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=dev_tokenized_dataset,
        processing_class=tokenizer
    )
    # train a model with specified arguments
    trainer.train()
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0)
    print('Training Done')
    # the below code is to save the most recent model
    model.save_pretrained(args.mod+'-final')
    predictions_dev = pipe(dev_tokenized_dataset['text'], truncation=True, max_length=256)
    # save the outputs on the dev and test datasets
    actual_labels_dev = []
    # to predict and return the class/label with the highest score
    for prediction in predictions_dev:
        pred_label = prediction['label']
        pred_index = pred_label.split('_')[1]
        actual_labels_dev.append(index_to_label_dict[int(pred_index)])
    write_lines_to_file(actual_labels_dev, args.o1)
    predictions_test = pipe(test_tokenized_dataset['text'], truncation=True, max_length=256)
    actual_labels_test = []
    for prediction in predictions_test:
        pred_label = prediction['label']
        pred_index = pred_label.split('_')[1]
        actual_labels_test.append(index_to_label_dict[int(pred_index)])
    write_lines_to_file(actual_labels_test, args.o2)


if __name__ == '__main__':
    main()
