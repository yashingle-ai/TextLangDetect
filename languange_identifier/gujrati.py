import re
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer

gujrati_text = """શિક્ષણ જીવનમાં ખૂબ મહત્વપૂર્ણ છે. તે માનવજાતને અજ્ઞાનતા અને ગેરસમજમાંથી બહાર લાવે છે. 
શિક્ષણ માત્ર પુસ્તક જ્ઞાન પૂરતું નથી, પરંતુ તે વ્યક્તિના વર્તન, વિચારસરણી અને જીવન જીવવાની રીતમાં પણ ફેરફાર લાવે છે. 
શિક્ષણના કારણે વ્યક્તિ નવા અવકાશમાં પ્રવેશ કરે છે અને નવી ક્ષમતાઓ વિકસાવે છે.

કોઈ પણ સમાજનો વિકાસ એના નાગરિકોના શિક્ષણની ગુણવત્તા પર આધાર રાખે છે. 
શિક્ષિત નાગરિકો તેમના દૈનિક જીવનમાં અને સમાજમાં મોટું યોગદાન આપી શકે છે. +૯૧ ૯૮૭૬૫૪૩૨૧૦
શિક્ષણ દરેક સ્તરે જરૂરી છે, તે શાળામાં હોય, કોલેજમાં હોય કે જીવનના અનુભવમાંથી હોય.

ગુજરાતના વૈભવી ઇતિહાસમાં શિક્ષણને હંમેશા મહત્વ આપવામાં આવ્યું છે. 
પ્રાચીન સમયમાં નલંદા અને તક્ષશિલા જેવા મહાન શૈક્ષણિક કેન્દ્રો ગુજરાતના અભિમાનના પ્રતીક છે. 
આજે પણ ગુજરાતમાં શિક્ષણ ક્ષેત્રે અનેક પ્રગતિ થઇ છે."""

# Correct Unicode ranges
gujarati_unicode_block = r'[\u0A80-\u0AFF]+'  # Gujarati script
gujarati_digits_block = r'[\u0AE6-\u0AEF]+'   # Gujarati digits

# 1. Word Tokenization
gujarati_word_pattern = r'[\u0A80-\u0AFF]+'
gujarati_words = re.findall(gujarati_word_pattern, gujrati_text)
print("Gujarati Words:", gujarati_words)

#1.1 paragraph splitter 
gujrati_paragraph=gujrati_text.split("/n/n")
# for paragraph in gujrati_paragraph:
#     print(paragraph.split("."))
# 2. Sentence Tokenization
gujarati_sentences = sent_tokenize(gujrati_text)
print("\nGujarati Sentences:")
for sentence in gujarati_sentences:
    print(sentence)

# 3. Email Extraction (for Gujarati + English-like emails)
email_pattern_gujarati = r'[\u0A80-\u0AFFa-zA-Z0-9._%+-]+@[a-zA-Z0-9._-]+\.[a-zA-Z]{2,}'
gujarati_emails = re.findall(email_pattern_gujarati, gujrati_text)
print("\nGujarati Emails:", gujarati_emails)

# 4. Mobile Number Extraction (Gujarati digits)
gujarati_mobile_pattern = r'(?:\+91\s?)?[\u0AE6-\u0AEF]{5}\s?[\u0AE6-\u0AEF]{5}'
gujarati_mobile_numbers = re.findall(gujarati_mobile_pattern, gujrati_text)
print("\nGujarati Mobile Numbers:", gujarati_mobile_numbers)

