#here we take two input files ,one is containing sentences and one is with thier unicode label 
#and we want to divide combine this sentence file and label file for test train and dev with the ratio of 80%,10%,10% 

import random
import argparse

#defining function for splitiing sentences 
def split_files(sentence_file,label_file,output_file_test_sentence,output_file_train_sentence,output_file_dev_sentence,output_file_train_unicodes,output_file_test_unicodes,output_file_dev_unicodes,test_ratio=0.1,train_ratio=0.8,dev_ratio=0.1):
    
    # The assert statement is used for debugging and to enforce conditions that must be true in the code.
    assert abs(test_ratio+train_ratio+dev_ratio-1)< 1e-6
    
    #reading files of sentences and labels(unicode of language like for english its eng)
    with open(sentence_file,'r',encoding='utf-8') as f_sentence:
        sentences=f_sentence.readlines()
    
    with open(label_file,'r',encoding='utf-8') as f_labels:
        labels=f_labels.readlines()
        
    #check if its not true then returns the error    
    assert len(sentences)==len(labels)
    
    #pairing sentence and thier unicode
    paired_data=list(zip(sentences,labels))
    
    # Setting the seed ensures that the random operations (like shuffling) can be repeated consistently.
    random.seed(42)
    
    
    # suffling the sentence with the use of random
    random.shuffle(paired_data)
    
    #splitiing the data
    total_samples=len(paired_data)
    train_end=int(total_samples*train_ratio)
    dev_end=int((total_samples*train_ratio) + (total_samples*dev_ratio))
    

    train_data = paired_data[:train_end]
    dev_data = paired_data[train_end:dev_end]
    test_data = paired_data[dev_end:]
    
    #unzip the data 
    train_sentence,train_labels=zip(*train_data)
    test_sentence,test_labels=zip(*test_data)
    dev_sentence,dev_labels=zip(*dev_data)
    
    #writing the sentence into files
    def write_into_file(file_path,data):
        with open(file_path,'w',encoding='utf-8') as f:
            f.writelines(data)
    
    #writing saparatly sentences for dev,test,train
    write_into_file(output_file_train_sentence,train_sentence)
    write_into_file(output_file_test_sentence,test_sentence)
    write_into_file(output_file_dev_sentence,dev_sentence)
    
    #writing saparatly unicodes for dev,test,train
    write_into_file(output_file_train_unicodes,train_labels)
    write_into_file(output_file_test_unicodes,test_labels)
    write_into_file(output_file_dev_unicodes,dev_labels)
    
    print("Files successfully created:")


    
#main function 

def main():
    parser=argparse.ArgumentParser(description='take 2 file input of sentence and thier unicode and split into test train dev text')
    parser.add_argument('input_file_sentence',type=str,help='enter the sentence contain text file path')
    parser.add_argument('input_file_unicode',type=str,help='enter the unicode contain text file path')
    parser.add_argument('output_file_test_sentence',type=str,help='enter the file path where you want to store the test sentence')
    parser.add_argument('output_file_train_sentence',type=str,help='enter the file path where you want to store the train sentece')
    parser.add_argument('output_file_dev_sentence',type=str,help='enter the file path where you want to store the dev sentece')
    parser.add_argument('output_file_train_unicodes',type=str,help='enter the file path where you want to store the train unicodes')
    parser.add_argument('output_file_test_unicodes',type=str,help='enter the file path where you want to store the test unicodes')
    parser.add_argument('output_file_dev_unicodes',type=str,help='enter the file path where you want to store the dev unicodes')
    args=parser.parse_args()

    #call split function 
    split_files(args.input_file_sentence,args.input_file_unicode,args.output_file_test_sentence,args.output_file_train_sentence,args.output_file_dev_sentence,args.output_file_train_unicodes,args.output_file_test_unicodes,args.output_file_dev_unicodes)
    
    
if __name__ == "__main__":
    main()

    
    
    
