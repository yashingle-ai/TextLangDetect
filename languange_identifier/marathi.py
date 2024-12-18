#for marathi language 

import re,nltk 
from nltk.tokenize import word_tokenize,sent_tokenize,RegexpTokenizer
from re import findall,U,search,findall
# marathi_text='''नाव: रोहन विजय पाटील  
# पत्ता: साई निवास, शिवाजी नगर, पुणे - ४११०१६  
# मोबाईल: +९१ ९८७६५ ४३२१०  
# ईमेल: रोहन.पाटील१२३@जीमेल.कॉम  
# जन्मतारीख: १५-०६-१९९५  
# लिंग: पुरुष  
# पेशा: सॉफ्टवेअर अभियंता  
# पिन: ४११०१६  

# व्यक्तिगत आवड:  
# संगीत ऐकणे  
# भटकंती करणे  
# वाचन आणि लेखन  
# क्रिकेट खेळणे
# '''
marathi_text="""एकदा एका जंगलात एक भलामोठा वटवृक्ष होता. त्या वटवृक्षाच्या छायेत अनेक पक्षी, प्राणी आणि कीटक राहत होते. त्या झाडावर एक छोटा चिमणा पक्षी राहत होता. त्याला नेहमी मोठ्या पक्ष्यांकडून टोमणे ऐकावे लागत. "तू किती छोटा आहेस, तू कुठे काही करू शकशील?" असे म्हणत त्याला हसायचे. 

पण एके दिवशी जंगलात भीषण वादळ आले. सर्व पक्षी घाबरले आणि झाडाच्या फांद्यांमध्ये लपून बसले. चिमणा मात्र घाबरला नाही. त्याने आपल्या चिमुकल्या चोचीने एका भेगातून पडणारे पाणी बाहेर काढायला सुरुवात केली. मोठे पक्षी त्याच्यावर हसत होते, पण चिमणा आपल्या कामात मग्न होता. 

थोड्या वेळाने वादळ शांत झाले. मोठ्या पक्ष्यांना समजले की त्या छोट्या चिमण्याने त्याच्या धाडसाने झाड वाचवले. त्या दिवसापासून सर्वांनी चिमण्याचा आदर करायला सुरुवात केली. 

कथानुकर्ष: 
कधीही स्वतःला कमी लेखू नका. आपली मेहनत आणि धाडस मोठ्या संकटांवर मात करू शकते.
"""
#for sentence tokenise and word tokenise

marathi_text_pattern=r"[\u0900-\u097F\w-]+"
paragraph=marathi_text.split("/n/n")
marathi_sentence_patern=r'.+?[।॥.]'
for para in paragraph:
    sentences=sent_tokenize(para)
    print(sentences)
    for sentence in sentences:
        regexp_tokensior=RegexpTokenizer(marathi_text_pattern)
        marathi_words=regexp_tokensior.tokenize(sentence)
        print(marathi_words)
        
#for word tokenize
words=findall(marathi_text_pattern,marathi_text)
print(words)
#finding email from the given input 
marathi_text_email=r'[\u0900-\u097F0-9._%+-]+@[\u0900-\u097Fa-zA-Z0-9.-]+\.[\u0900-\u097Fa-zA-Z]{2,}'
marathi_email=findall(marathi_text_email,marathi_text)
print(marathi_email)

#for finding mobile numbers 
marathi_text_mobile=r'\+?[०-९]{2}[\s]?[०-९]{5}[\s]?[०-९]{5}'
mobile_no=findall(marathi_text_mobile,marathi_text)
print(mobile_no)

#in marathi sentence ends with | || .