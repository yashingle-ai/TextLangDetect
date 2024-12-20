# Language Tagging in sentence
# Import required libraries
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

# Define a class for language identification
class LanguageIdentifier:
    """
    A class to identify the language of text based on Unicode character patterns.
    """

    def __init__(self):
        # Mapping of language codes to their respective Unicode character ranges.
        self.language_patterns = {
            "hin": r'[\u0900-\u097F]+',  # Hindi
            "ben": r'[\u0980-\u09FF]+',  # Bengali
            "pan": r'[\u0A00-\u0A7F]+',  # Punjabi
            "urd": r'[\u0600-\u06FF]+',  # Urdu
            "ori": r'[\u0B00-\u0B7F]+',  # Odia
            "mar": r'[\u0900-\u097F]+',  # Marathi
            "mal": r'[\u0D00-\u0D7F]+',  # Malayalam
            "tam": r'[\u0B80-\u0BFF]+',  # Tamil
            "tel": r'[\u0C00-\u0C7F]+',  # Telugu
            "guj": r'[\u0A80-\u0AFF]+',  # Gujarati
            "kan": r'[\u0C80-\u0CFF]+',  # Kannada
            "asm": r'[\u0980-\u09FF]+',  # Assamese
            "kok": r'[\u0900-\u097F]+',  # Konkani
            "eng": r'[a-zA-Z0-9]+'      # English
        }

    def identify_language(self, text):
        """
        Identify the language of each sentence in the input text.

        Args:
            text (str): The input text.

        Returns:
            None
        """
        
        # Dictionary to store language counts (optional for additional analysis)
        language_count = {}

        # Tokenize the text into sentences
        sentences = sent_tokenize(text)

        # Iterate through each sentence
        for sentence in sentences:
            language_detected = False  # Flag to check if a language is identified
            
            # Check the sentence against each language pattern
            for language_code, pattern in self.language_patterns.items():
                # Find matches for the language pattern in the sentence
                matched_words = re.findall(pattern, sentence)

                if matched_words:
                    # Print the sentence with its identified language tag
                    print(f'{sentence}\t{language_code}')
                    language_detected = True
                    break  # Stop checking further patterns once a language is identified
            
            # If no language is detected, mark it as unidentified
            if not language_detected:
                print(f'{sentence}\tUnidentified')

# Create an instance of LanguageIdentifier
language_identifier = LanguageIdentifier()

# Input text containing multiple languages
input_text = (
    "my name is yash ingle and I don't want to do anything from your side. "
    "ಮಾನವಜಾತಿ ಎದುರಿಸುತ್ತಿರುವ ಪ್ರಮುಖ ಸಮಸ್ಯೆಗಳಿಗೆ ಪರಿಹಾರವನ್ನು ಶಿಕ್ಷಣದ ಮೂಲಕವೇ ಕಂಡುಹಿಡಿಯಬಹುದಾಗಿದೆ. "
    "तो राजा हमारी गलियों में नहीं आते।"
)
# Input text for language tagging
text = """
my name is novel and i am reading book. 
DiwaliThe Festival of Lights
Diwali, also known as Deepavali, is one of the most significant and widely celebrated festivals in India. 
It is often referred to as the "Festival of Lights" because of the beautiful display of lamps and candles that 
illuminate homes and public spaces during this time. The festival signifies the triumph of light over darkness 
and good over evil, making it a time of joy, hope, and togetherness.
Historical Significance
The origins of Diwali can be traced to several stories from Indian mythology. 
The most popular legend is the return of Lord Rama to Ayodhya after 14 years of exile, 
during which he defeated the demon king Ravana. The people of Ayodhya celebrated his return by lighting oil lamps 
(diyas), symbolizing the victory of good over evil. Another important story is that of Lord Krishna's victory 
over the demon Narakasura, which also signifies the eradication of darkness.
"""

# Call the language identification method
language_identifier.identify_language(text)
