import re
import nltk
from re import search
from nltk.tokenize import sent_tokenize, RegexpTokenizer

odia_2 = """ନାମ: ରାଜେଶ କୁମାର ଦାସ
ଠିକଣା: ମାଇତ୍ରୀ ବିହାର, ଭୁବନେଶ୍ୱର, ଓଡିଶା - ୭୫୧୦୦୨
ମୋବାଇଲ: +୯୧ ୯୮୫୬୭ ୧୨୩୪୫
ଇମେଲ: ରାଜେଶ.ଦାସ୧୨୩@ଜିମେଲ.କମ୍
ଜନ୍ମତାରିଖ: ୧୫-୦୬-୧୯୯୫
ଲିଙ୍ଗ: ପୁରୁଷ
ପେଶା: ଯନ୍ତ୍ରିକ ଇଂଜିନିୟର
ପିନ୍: ୭୫୧୦୦୨

ବ୍ୟକ୍ତିଗତ ରୁଚି:

ସଂଗୀତ ସୁଣିବା
ଭ୍ରମଣ କରିବା
ବିଜ୍ଞାନ ଗବେଷଣା ପଢିବା"""

# Define Odia Unicode block
odia_unicode_block = r'[\u0B00-\u0B7F\u200C\u200D\w]+'

# Split the paragraphs by double newline
paragraphs = odia_2.split("\n\n")

# Process each paragraph
for para in paragraphs:
    sentences = sent_tokenize(para)  # Tokenize sentences from the paragraph
    for sentence in sentences:
        # Create a regular expression tokenizer for words
        regexp_tokenizer = RegexpTokenizer(odia_unicode_block)
        words = regexp_tokenizer.tokenize(sentence)  # Tokenize words
        print(words)  # Print the word tokens

        # Only print sentences that contain Odia Unicode characters
        if re.search(odia_unicode_block, sentence):
            print("Odia Sentence:", sentence)

#for mobile_number in the given text 
odia_mobile=r'\+?[୦-୯]{2}[\s]?[୦-୯]{5}[\s]?[୦-୯]{5}'
mobile_number=re.findall(odia_mobile,odia_2)
print("mobile_number from the given text in odia language is : ",mobile_number)
