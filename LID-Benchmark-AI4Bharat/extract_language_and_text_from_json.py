"""Extract text and language from JSON."""
from argparse import ArgumentParser
from json import load
from re import sub


full_to_short_lang_code_mapping = {
    "Assamese": "asm",
    "Bangla": "ben",
    "Bodo": "brx",
    "Dogri": "doi",
    "Gujarati": "guj",
    "Hindi": "hin",
    "Kannada": "kan",
    "Kashmiri": "kas",
    "Konkani": "gom",
    "Maithili": "mai",
    "Malayalam": "mal",
    "Manipuri": "mni",
    "Marathi": "mar",
    "Nepali": "npi",
    "Oriya": "ory",
    "Punjabi": "pan",
    "Sanskrit": "san",
    "Santali": "sat",
    "Sindhi": "snd",
    "Tamil": "tam",
    "Telugu": "tel",
    "Urdu": "urd"
}

def load_object_from_json(json_file):
    """Load a python object from a JSON file."""
    with open(json_file, 'r', encoding='utf-8', errors='ignore') as json_load:
        return load(json_load)


def write_line_to_file(line, file_path):
    """Write a single line to file."""
    with open(file_path, 'a', encoding='utf-8') as file_write:
        file_write.write(line + '\n')


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser()
    parser.add_argument('--input', dest='inp', help='Enter the input JSON file.')
    parser.add_argument('--output', dest='out', help='Enter the output text file.')
    parser.add_argument('--skip', dest='skip', help='Enter the language to skip.', default='')
    args = parser.parse_args()
    dict_loaded = load_object_from_json(args.inp)
    list_of_dicts = dict_loaded['data']
    for d in list_of_dicts:
        sentence_text = d["native sentence"]
        language = d["language"]
        # print(sentence_text, language)
        if sentence_text and language:
            sentence_text = sub('\\s{2,}', ' ', sentence_text)
            sentence_text = sentence_text.replace('\t', ' ')
            if args.skip:
                if language != args.skip:
                    line_to_write = sentence_text + '\t' + full_to_short_lang_code_mapping[language]
                    write_line_to_file(line_to_write, args.out)
            else:
                line_to_write = sentence_text + '\t' + full_to_short_lang_code_mapping[language]
                write_line_to_file(line_to_write, args.out)
        else:
            print('NOT VALID', language, sentence_text)


if __name__ == '__main__':
    main()
