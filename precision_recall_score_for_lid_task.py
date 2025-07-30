"""Find Precision, Recall, F1 Score with classification_report for the Language Identification task."""
from argparse import ArgumentParser
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def find_precision_recall_f1_score(gold_labels, predicted_labels, true_labels=None):
    return classification_report(gold_labels,
                                 predicted_labels, target_names=true_labels)


def main():
    parser = ArgumentParser(description='This program is about finding precision, recall, and F1 score for the Language Identification task.')
    parser.add_argument('--gold', dest='g', help='Enter the gold standard file path.')
    parser.add_argument('--pred', dest='p', help='Enter the predicted file path.')
    parser.add_argument('--output', dest='o', help='Enter the output file path for the classification report.')
    args = parser.parse_args()
    gold_path = args.g
    pred_path = args.p
    out_path = args.o
    gold = read_lines_from_file(gold_path)
    predicted = read_lines_from_file(pred_path)
    all_labels = set(predicted).union(set(gold))
    dict_label_to_indices = {label: index for index, label in enumerate(all_labels)}
    predicted_into_indexes = [dict_label_to_indices[item] for item in predicted]
    gold_into_indexes = [dict_label_to_indices[item] for item in gold]
    out_desc = open(out_path, 'w')
    class_report = ''
    class_report += find_precision_recall_f1_score(gold, predicted)
    if len(set(predicted_into_indexes)) == 2:
        print('Micro Precision =', precision_score(gold_into_indexes, predicted_into_indexes, average='binary'))
        print('Micro Recall =', recall_score(gold_into_indexes, predicted_into_indexes, average='binary'))
        print('Micro F1 =', f1_score(gold_into_indexes, predicted_into_indexes, average='binary'))
        print('Micro Accuracy =', accuracy_score(gold_into_indexes, predicted_into_indexes))
    else:
        class_report += '\n'
        class_report += 'Micro_Precision = ' + str(precision_score(gold_into_indexes, predicted_into_indexes, average='micro')) + '\n'
        print('Micro Precision =', precision_score(gold_into_indexes, predicted_into_indexes, average='micro'))
        class_report += 'Micro_Recall = ' + str(recall_score(gold_into_indexes, predicted_into_indexes, average='micro')) + '\n'
        print('Micro Recall =', recall_score(gold_into_indexes, predicted_into_indexes, average='micro'))
        class_report += 'Micro_F1 = ' + str(f1_score(gold_into_indexes, predicted_into_indexes, average='micro')) + '\n'
        print('Micro F1 =', f1_score(gold_into_indexes, predicted_into_indexes, average='micro'))
        class_report += 'Micro_Accuracy = ' + str(accuracy_score(gold_into_indexes, predicted_into_indexes)) + '\n'
        print('Micro Accuracy =', accuracy_score(gold_into_indexes, predicted_into_indexes))
    out_desc.write(class_report + '\n')
    out_desc.close()


if __name__ == '__main__':
    main()
