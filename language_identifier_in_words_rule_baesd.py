#language identifier rule based for 14 indian languages based on thier unicodes for words 
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

    def identify_language_of_words(self, text):
        '''
        Identify the dominant language in the given text based on character patterns.

        Args:
            text (str): The input text for language identification.

        Returns:
            str: The name of the language with thier words .
        '''
        
        print("Length of given text is", len(text))

        # Dictionary to store the total matched character length for each language.
        words_with_thier_language = []

        # Check for empty input text.
        if len(text) == 0:
            print("No input provided for identification.")
            return None

        # Iterate through each language and its pattern.
        for word in text.split():
            for language, pattern in self.language_patterns.items():
            # Find all occurrences of the pattern in the text.
                matched_words = re.findall(pattern,word)
                # print(matched_words)
                if matched_words:
                # now i am inserting the word and thier matched language as 1 element of list 
                   words_with_thier_language.append((matched_words,language))
                
                
                         
            

        #return the list that contain words with thier particular matched list 
        return words_with_thier_language

# Instantiate the class.
language_identifier = LanguageIdentifier()

# Input text to identify language.
input_text = (
    "my name is yash ingle and i dont want to do anthing from your side  "
    "ಮಾನವಜಾತಿ ಎದುರಿಸುತ್ತಿರುವ ಪ್ರಮುಖ ಸಮಸ್ಯೆಗಳಿಗೆ ಪರಿಹಾರವನ್ನು ಶಿಕ್ಷಣದ ಮೂಲಕವೇ ಕಂಡುಹಿಡಿಯಬಹುದಾಗಿದೆ. "
    "तो राजा हमारी गलियों में नहीं आते"
)

#calling the funtion

list_of_words_with_thier_language=language_identifier.identify_language_of_words(input_text)
# Print the identified words 
print("The dominant language is:",list_of_words_with_thier_language)
