# #for kannada language 
# import re,nltk 
# from nltk.tokenize import word_tokenize,sent_tokenize,RegexpTokenizer
# from re import findall,U,search,findall

# kannada_text='''ಶಿಕ್ಷಣವು ಮಾನವನ ಬದುಕಿನಲ್ಲಿ ಮಹತ್ವದ ಪಾತ್ರ ವಹಿಸುತ್ತದೆ. ಇದು ವ್ಯಕ್ತಿಯ ಜ್ಞಾನವನ್ನು ವಿಸ್ತರಿಸುವ ಜೊತೆಗೆ, ಬುದ್ಧಿಮತ್ತೆ ಮತ್ತು ನಿರ್ಣಯ ಸಾಮರ್ಥ್ಯವನ್ನು ಹೆಚ್ಚಿಸುತ್ತದೆ. ಶಿಕ್ಷಣವು ವ್ಯಕ್ತಿಯನ್ನು ಸಮಾಜದ ಜವಾಬ್ದಾರಿಯುತ ನಾಗರಿಕನಾಗಿಸುತ್ತದೆ. ಶಾಲಾ ಶಿಕ್ಷಣ ಮಕ್ಕಳಲ್ಲಿ ಶಿಸ್ತಿನ ಅರಿವು ಮೂಡಿಸುತ್ತದೆ ಮತ್ತು ಮೌಲ್ಯಾಧಾರಿತ ಜೀವನಕ್ಕಾಗಿ ಬುನಾದಿ ರೂಪಿಸುತ್ತದೆ.

# ಒಬ್ಬ ವ್ಯಕ್ತಿ ಶಿಕ್ಷಣದ ಮೂಲಕ ಹೊಸ ವಿಶ್ವವನ್ನು ಕಂಡುಕೊಳ್ಳುತ್ತಾನೆ. ಪುಸ್ತಕಗಳ ಜ್ಞಾನ ಮತ್ತು ಅನುಭವಗಳ ಸಮನ್ವಯದಿಂದ ಬದುಕಿನ ಹೊಸ ಆಯಾಮವನ್ನು ಅನುಭವಿಸಬಹುದು. ಶಿಕ್ಷಣವು ಕೇವಲ ಉದ್ಯೋಗ ಪಡೆಯುವ ಸಾಧನ ಮಾತ್ರವಲ್ಲ, ಅದು ಜೀವನದ ಉತ್ತಮ ಗುಣಮಟ್ಟಕ್ಕಾಗಿ ಅನಿವಾರ್ಯವಾದ ಮಾರ್ಗವಾಗಿದೆ. ರಾಜೇಶ್.ಕುಮಾರ್@ಗ್ಮೇಲ್.ಕಾಂ  

# ಮಾನವಜಾತಿ ಎದುರಿಸುತ್ತಿರುವ ಪ್ರಮುಖ ಸಮಸ್ಯೆಗಳಿಗೆ ಪರಿಹಾರವನ್ನು ಶಿಕ್ಷಣದ ಮೂಲಕವೇ ಕಂಡುಹಿಡಿಯಬಹುದಾಗಿದೆ. ಶಿಕ್ಷಣದ ಮಹತ್ವವನ್ನು ಅರಿತು, ಸರಕಾರಗಳು ಮತ್ತು ಸಂಸ್ಥೆಗಳು ಉತ್ತಮ ಗುಣಮಟ್ಟದ ಶಿಕ್ಷಣವನ್ನು ಎಲ್ಲರಿಗೂ ಒದಗಿಸಲು ಪ್ರಯತ್ನಿಸಬೇಕು. +೯೧ ೯೮೫೬೭ ೧೨೩೪೫  7828110014 +91 7828110014

# ಪ್ರಾಚೀನ ಭಾರತದಲ್ಲಿ ಶಿಕ್ಷಣಕ್ಕೆ ಅತ್ಯಂತ ಮಹತ್ವವಿತ್ತು. ನಲಂದಾ, ತಕ್ಷಶಿಲಾ ಇಂತಹ ಶೈಕ್ಷಣಿಕ ಕೇಂದ್ರಗಳು ಜಗತ್ತಿನ ಮಟ್ಟಿಗೆ ಪ್ರಸಿದ್ಧವಾಗಿದ್ದವು. ಇಂದಿನ ಯುಗದಲ್ಲಿ ವಿಜ್ಞಾನ ಮತ್ತು ತಂತ್ರಜ್ಞಾನವು ಶಿಕ್ಷಣ ವ್ಯವಸ್ಥೆಗೆ ಹೊಸ ಆಯಾಮವನ್ನು ನೀಡುತ್ತಿದೆ. ಶಿಕ್ಷಣದ ಕ್ಷೇತ್ರದಲ್ಲಿ ನಿರಂತರ ಅಭಿವೃದ್ಧಿ ಮತ್ತು ಹೊಸ ಹೊಸ ಪ್ರಯೋಗಗಳು ನಡೆಯುತ್ತಿವೆ.'''

# # The range \u0C80-\u0CFF includes all Kannada letters (vowels, consonants, diacritics, etc.).
# # The range \u0CE6-\u0CEF includes Kannada numerals (0-9).
# kannada_unicode_block=r'[\u0C80-\u0CFF]' 
# kannada_digit_block=r'[\u0CE6-\u0CEF]'

# # 1. Word Tokenization
# kannada_word_pattern = r'[\u0C80-\u0CFF]+'
# kannada_words = re.findall(kannada_word_pattern, kannada_text)
# print("kannada Words:", kannada_words)

# #2. kannada sentence tokensiation  mainly sentences are ends with . and somtimes with ?!comma are not used to separate the sentences 
# kannada_sentence_pattern=r'[\u0C80-\u0CFF\s]+[.!?]'
# kannada_sentence=re.findall(kannada_sentence_pattern,kannada_text)
# print("kannada sentence is :-  ",kannada_sentence)

# #3.kannada paragraph toenisation
# kannada_paragraph=kannada_text.split("/n/n")
# print(kannada_paragraph)

# # 3. Email Extraction (for kannada + English-like emails)
# email_pattern_kannada = r'[\u0C80-\u0CFFa-zA-Z0-9._%+-]+@[\u0C80-\u0CFFa-zA-Z0-9._-]+\.[\u0C80-\u0CFFa-zA-Z]{2,}'
# kannada_emails = re.findall(email_pattern_kannada, kannada_text)
# print("\nkannada Emails:", kannada_emails)

# # 4. Mobile Number Extraction (kannada digits)
# kannada_mobile_pattern = r'(?:\+91\s?)?(?:[\+\u0CE6-\u0CEF]{2}\s?)?[\u0CE6-\u0CEF0-9]{5}\s?[\u0CE6-\u0CEF0-9]{5}'
# kannada_mobile_numbers = re.findall(kannada_mobile_pattern, kannada_text)
# print("\nkannada Mobile Numbers:", kannada_mobile_numbers)


