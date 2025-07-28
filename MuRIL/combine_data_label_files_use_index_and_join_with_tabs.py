"""Combine data and labels as indexes and join each line with tabs."""
import os
from argparse import ArgumentParser
from re import sub
from pickle import dump


def dump_object_into_pickle(data_object, pickle_file):
    """
    Dump a python object into a pickle file.

    Args:
    data_object: Data object to be pickled.
    pickle_file: Enter the path of the pickle file.

    Returns: None
    """
    with open(pickle_file, 'wb') as file_dump:
        dump(data_object, file_dump)


def read_lines_from_file(file_path):
    """
    Read lines from a file.

    Args:
    file_path: Path to the input file.

    Returns:
    lines: List of lines read from the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file_read:
        lines = [line.strip() for line in file_read.readlines() if line.strip()]
        updated_lines = []
        for line in lines:
            # Remove any leading or trailing whitespace and replace multiple spaces with a single space
            line = sub(r'\s+', ' ', line)
            line = line.strip()
            updated_lines.append(line)
        return updated_lines


def write_lines_to_file(lines, file_path):
    """
    Write lines to a file.

    Args:
    lines: Lines to be written to the file.
    file_path: Path to the output file.

    Returns: None
    """
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines))


def combine_files_and_join_lines_with_tabs(data_lines, label_lines, output_file):
    """
    Combine files and join each line with tabs.

    Args:
    file_paths: List of file paths to be combined.
    output_file: Path to the output file.

    Returns: None
    """
    combined_lines = ['text\tlabel']
    for index, data_line in enumerate(data_lines):
        if index < len(label_lines):
            label_line = label_lines[index]
            # Join data and label lines with a tab
            combined_line = f"{data_line}\t{label_line}"
            combined_lines.append(combined_line)
        else:
            # If there are more data lines than label lines, just append the data line
            combined_lines.append(data_line)

    write_lines_to_file(combined_lines, output_file)


def main(): 
    """
    Main function to parse arguments and call the combine function.

    Args: None

    Returns: None
    """
    parser = ArgumentParser(description='Combine data and label files with labels as indexes, join them with tabs, and write to an output file.')
    parser.add_argument('--data', dest='data', help='Enter the data file path')
    parser.add_argument('--label', dest='label', help='Enter the label file path')
    parser.add_argument('--output', dest='out', help='Output file path.')
    parser.add_argument('--pickle', dest='pickle', help='Path to save the index-to-label dictionary as a pickle file.')
    args = parser.parse_args()
    data_lines = read_lines_from_file(args.data)
    label_lines = read_lines_from_file(args.label)
    if not data_lines or not label_lines:
        print("Data or label file is empty. Please provide valid files.")
        return
    if len(data_lines) != len(label_lines):
        print("Data and label files have different lengths. Please check the files.")
        return
    unique_labels = set(label_lines)
    if len(unique_labels) < 2:
        print("Label file must contain at least two unique labels for classification.")
        return
    unique_labels = sorted(unique_labels)  # Sort labels for consistent indexing
    # Create a dictionary mapping labels to indices
    index_to_label_dict = {index: label for index, label in enumerate(unique_labels)}
    label_to_index_dict = {label: index for index, label in index_to_label_dict.items()}
    # Convert labels to indices
    print(f"Label to index mapping: {label_to_index_dict}")
    print(f"Index to label mapping: {index_to_label_dict}")
    # Convert label lines to indices
    labels_as_indices = [label_to_index_dict[label] for label in label_lines]
    combine_files_and_join_lines_with_tabs(data_lines, labels_as_indices, args.out)
    print(f"Combined data and labels written to {args.out}")
    dump_object_into_pickle(index_to_label_dict, args.pickle)


if __name__ == "__main__":
    main()
    # This allows the script to be run directly from the command line.
    # It will parse the command line arguments and call the main function.
