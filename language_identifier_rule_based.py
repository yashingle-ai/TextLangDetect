#language identifier rule based for 14 indian languages based on thier unicodes
import re

class LanguageIdentifier:
    '''
    Language identification for texts based on Unicode character patterns.
    '''

    def __init__(self):
        # Mapping of languages to their respective Unicode character ranges.
        self.language_patterns = {
            "Hindi": r'[\u0900-\u097F]+',
            "Bengali": r'[\u0980-\u09FF]+',
            "Punjabi": r'[\u0A00-\u0A7F]+',
            "Urdu": r'[\u0600-\u06FF]+',
            "Odia": r'[\u0B00-\u0B7F]+',
            "Marathi": r'[\u0900-\u097F]+',
            "Malayalam": r'[\u0D00-\u0D7F]+',
            "Tamil": r'[\u0B80-\u0BFF]+',
            "Telugu": r'[\u0C00-\u0C7F]+',
            "Gujarati": r'[\u0A80-\u0AFF]+',
            "Kannada": r'[\u0C80-\u0CFF]+',
            "Assamese": r'[\u0980-\u09FF]+',
            "Konkani": r'[\u0900-\u097F]+',
            "English": r'[a-zA-Z0-9]+'
        }

    def identify_language(self, text):
        '''
        Identify the dominant language in the given text based on character patterns.

        Args:
            text (str): The input text for language identification.

        Returns:
            str: The name of the language with the maximum character count.
        '''
        
        print("Length of given text is", len(text))

        # Dictionary to store the total matched character length for each language.
        language_character_count = {}

        # Check for empty input text.
        if len(text) == 0:
            print("No input provided for identification.")
            return None

        # Iterate through each language and its pattern.
        for language, pattern in self.language_patterns.items():
            # Find all occurrences of the pattern in the text.
            matched_words = re.findall(pattern, text)

            # Calculate the total length of matched characters for the language.
            total_length = sum(len(word) for word in matched_words)

            # Store the total length in the dictionary.
            language_character_count[language] = total_length

        # Find the language with the highest character count without lambda.
        max_occurrence_language = None
        max_count = 0

        for language, count in language_character_count.items():
            if count > max_count:
                max_occurrence_language = language
                max_count = count

        return max_occurrence_language

# Instantiate the class.
language_identifier = LanguageIdentifier()

# Input text to identify language.
input_text = (
    "my name is yash ingle and i dont want to do anthing from your side  "
    "ಮಾನವಜಾತಿ ಎದುರಿಸುತ್ತಿರುವ ಪ್ರಮುಖ ಸಮಸ್ಯೆಗಳಿಗೆ ಪರಿಹಾರವನ್ನು ಶಿಕ್ಷಣದ ಮೂಲಕವೇ ಕಂಡುಹಿಡಿಯಬಹುದಾಗಿದೆ. "
    "तो राजा हमारी गलियों में नहीं आते"
)

# Call the language identification method.
dominant_language = language_identifier.identify_language(input_text)

# Print the identified dominant language.
print("The dominant language is:", dominant_language)
