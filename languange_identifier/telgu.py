#for telgu language 
import re,nltk 
from nltk.tokenize import word_tokenize,sent_tokenize,RegexpTokenizer
from re import findall,U,search,findall

telgu_text="""ఒక గ్రామంలో ఒక పేద కూలీ జీవించేవాడు. అతనికి ఒక చిన్న కుటుంబం ఉంది – భార్య, ఇద్దరు పిల్లలు. పేదరికం అతని జీవితానికి పెద్ద సవాలు. అతను రోజుకు కష్టపడుతూ కొన్ని రూపాయలు సంపాదించేవాడు. అయినా తన కుటుంబానికి సరిపడా ఆహారం తెచ్చుకునే స్థితిలో ఉండేవాడు. 

ఒక రోజు, అతను కూలి పనికి వెళ్లినపుడు, రోడ్డుపై ఒక నాణెం కనుగొన్నాడు. నాణెం చిన్నదే అయినా, అది అతని జీవితాన్ని మార్చేలా అనిపించింది. ఆ నాణెం తీసుకుని అతను మొదట ఏం చేయాలో ఆలోచించాడు. కొంత సేపు ఆలోచించి, ఆ డబ్బుతో తన పిల్లలకు పుస్తకాలు కొనాలని నిర్ణయించుకున్నాడు. 

తన పిల్లలు చదువు కొంటే, వారు పెద్దలు కావచ్చని అతని ఆశ. ఆ రోజు నుంచి అతను మరింత కష్టపడి పని చేసేవాడు. పిల్లల విద్య కోసం నిత్యం త్యాగాలు చేస్తూ, వారిని పాఠశాలకు పంపేవాడు. yash.ingle003@gmail.com

కాలం గడిచే కొద్దీ, అతని పిల్లలు చదువులో మెరుగ్గా ఎదిగారు. ఒకరు డాక్టర్ అయ్యారు, మరొకరు ఇంజనీర్ అయ్యారు. తన పిల్లల విజయాన్ని చూసినప్పుడు, ఆ పేద కూలీ గర్వంగా తల ఎత్తుకున్నాడు. అతని కష్టం, త్యాగం విలువైన ఫలితాన్ని అందించింది.

### నీతి: ౦౩౮౯౯౭౪౨౩౬ +91 ౯౮౮౭౨౪౩౨౧౦ రాజేష్.దాస్123@జీమెయిల్.కామ్ 
తల్లిదండ్రుల కష్టం మరియు పిల్లల విద్య కష్టకాలాలను జయించగలదు. ధనం కాదు, జ్ఞానం నిజమైన సంపద.
"""
telgu_text_patern=r'[\u0C00-\u0C7F]+'
telgu_sentence_pattern=r'.+?[।.!?॥]'    # telgu sentence ends with | || . ? !
telgu_paragraph=telgu_text.split("/n/n")

#for telgu senteces
for  para in telgu_paragraph:
    telgu_sentence=findall(telgu_sentence_pattern,para)
    print(telgu_sentence)
print("sentence ends ")

#for telgu words 
for para in telgu_paragraph:
    telgu_words=findall(telgu_text_patern,telgu_text)
    print(telgu_words)
    
#for mobile number in the given text 
# Regex pattern to match Telugu mobile numbers
# mobile_pattern_telugu = r'\+91[\s-]?[౭-౯]\d{4}[\s-]?\d{5}|\(?0\)?[\s-]?[౭-౯]\d{4}[\s-]?\d{5}'
mobile_pattern_telugu=r'[౦౧౨౩౪౫౬౭౮౯]{10} |\+91[\s-]?[౦౧౨౩౪౫౬౭౮౯]{10}  | \+91[\s-]?[౦౧౨౩౪౫౬౭౮౯]{5}[\s-]?[౦౧౨౩౪౫౬౭౮౯]{5}'

telgu_mobile=findall(mobile_pattern_telugu,telgu_text)
print("mobile number from telgu text  is : ",telgu_mobile)

#for telgu emails 
telgu_email_pattern=r'[\u0C00-\u0C7Fa-zA-Z0-9._]+@[\u0C00-\u0C7Fa-zA-Z0-9._]+.[\u0C00-\u0C7Fa-zA-Z0-9._]+| [a-zA-Z0-9._]+@[a-zA-Z0-9._].[a-zA-Z0-9._]{2,} '
telgu_email=findall(telgu_email_pattern,telgu_text)
print("email from telgu text is :- ",telgu_email)
