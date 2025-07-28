"""This code tokenizes text in various Indian languages without sentence tokenization.
It reads input files, tokenizes the content, and writes the output to specified files.
It supports both file and folder inputs, processing each file individually."""
# how to run the code
# python3 tokenizer_for_all_indian_languages_in_raw_format_without_tokenization.py --input Input --output Output
# works at folder and file levels
import re
import argparse
import os
from string import punctuation


# the below code defines different kinds of regular expressions
token_specification = [
    ('datemonth',
     r'^(0?[1-9]|1[012])[-\/\.](0?[1-9]|[12][0-9]|3[01])[-\/\.](1|2)\d\d\d$'),
    ('monthdate',
     r'^(0?[1-9]|[12][0-9]|3[01])[-\/\.](0?[1-9]|1[012])[-\/\.](1|2)\d\d\d$'),
    ('yearmonth',
     r'^((1|2)\d\d\d)[-\/\.](0?[1-9]|1[012])[-\/\.](0?[1-9]|[12][0-9]|3[01])'),
    ('EMAIL1', r'([\w\.])+@(\w)+\.(com|org|co\.in)$'),
    ('url1', r'(www\.)([-a-z0-9]+\.)*([-a-z0-9]+.*)(\/[-a-z0-9]+)*/i'),
    ('url', r'/((?:https?\:\/\/|www\.)(?:[-a-z0-9]+\.)*[-a-z0-9]+.*)/i'),
    ('BRACKET', r'[\(\)\[\]\{\}]'),       # Brackets
    ('NUMBER', r'^(\d+)([,\.۔]\d*)*(\w)*'),  # Integer or decimal number
    ('ASSIGN', r'[~:]'),          # Assignment operator
    ('END', r'[;!_]'),           # Statement terminator
    ('EQUAL', r'='),   # Equals
    ('OP', r'[+*\/\-]'),    # Arithmetic operators
    ('QUOTES', r'[\"\'‘’“”]'),          # quotes
    ('Fullstop', r'(\.+)$'),
    ('ellips', r'\.(\.)+'),
    ('HYPHEN', r'[-+\|+]'),
    ('Slashes', r'[\\\/]'),
    ('COMMA12', r'[,%]'),
    ('hin_stop', r'।'),
    ('urd_stop', r'۔'),
    ('urd_question_mark', r'؟'),
    ('quotes_question', r'[”\?]'),
    ('hashtag', r'#')
]
# the below code converts the above expression into a python regex
tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
get_token = re.compile(tok_regex)
punctuations = punctuation + '\"\'‘’“”'


def tokenize(list_s):
    """Tokenize a list of tokens."""
    tkns = []
    for wrds in list_s:
        wrds_len = len(wrds)
        initial_pos = 0
        end_pos = 0
        while initial_pos <= (wrds_len - 1):
            mo = get_token.match(wrds, initial_pos)
            if mo is not None and len(mo.group(0)) == wrds_len:
                tkns.append(wrds)
                initial_pos = wrds_len
            else:
                match_out = get_token.search(wrds, initial_pos)
                if match_out is not None:
                    end_pos = match_out.end()
                    if match_out.lastgroup == "NUMBER":
                        aa = wrds[initial_pos:(end_pos)]
                    else:
                        aa = wrds[initial_pos:(end_pos - 1)]
                    if aa != '':
                        tkns.append(aa)
                    if match_out.lastgroup != "NUMBER":
                        tkns.append(match_out.group(0))
                    initial_pos = end_pos
                else:
                    tkns.append(wrds[initial_pos:])
                    initial_pos = wrds_len
    return tkns


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def read_file_and_tokenize(input_file):
    """Read a file and tokenize its content by specifying the input file path and language type."""
    file_read = open(input_file, 'r', encoding='utf-8')
    lines = read_lines_from_file(input_file)
    sentences = lines
    proper_sentences = []
    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if sentence:
            list_tokens = tokenize(sentence.split())
            proper_sentences.append(' '.join(list_tokens))
    return proper_sentences


def write_list_to_file(output_file, data_list):
    """Write a list to a file."""
    with open(output_file, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(data_list) + '\n')


def main():
    """Pass arguments and call functions here."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', dest='inp', help="enter the input file path")
    parser.add_argument(
        '--output', dest='out', help="enter the output file path")
    args = parser.parse_args()
    if os.path.isdir(args.inp) and not os.path.isdir(args.out):
        os.mkdir(args.out)
    if not os.path.isdir(args.inp):
        sentences = read_file_and_tokenize(args.inp)
        write_list_to_file(args.out, sentences)
    else:
        for root, dirs, files in os.walk(args.inp):
            for fl in files:
                input_file_path = os.path.join(root, fl)
                sentences = read_file_and_tokenize(input_file_path)
                output_file_path = os.path.join(args.out, fl)
                write_list_to_file(output_file_path, sentences)


if __name__ == '__main__':
    main()
