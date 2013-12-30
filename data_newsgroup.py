# coding: gbk
import os
import glob
import sys
from doc import *

STOP_WORDS_SET = set()

#author: Wayne Lin
#deal with input [./data/20_newsgroups/]

class NewsGroupDocument(Document):

    def __init__(self, filepath):
        Document.__init__(self, filepath)
        #use filepath for label
        label=self.filepath[self.filepath.rfind('/',0,-10)+1:self.filepath.rfind('/')]      
        self.label = label 

    def split(self, STOP_WORDS_SET):
        '''
        split 20_newsgroups and ignore these head text
            "
            Xref: 
            history
            Newsgroups: 
            Path: 
            From: 
            Subject: Re: 
            Lines: 15
        
            "
        '''
        has_Lines=False
        for line in self.file:
            line=line.strip()
            if line=='':
                continue
            if 'Lines: ' in line:
                has_Lines=True
                continue
            if has_Lines:
                self.lines.append(line)
        for line in self.lines:
#            print line
            words = line.split(' ')
            for word in words:
                clean_word = self._clean_word(word)
                if clean_word and (clean_word not in STOP_WORDS_SET) and (len(clean_word) > 1): # omit stop words
                    self.words.append(clean_word)
        
        del self.lines
        gc.collect()
        self.file.close()

def getData(train_path,test_path):    
    # load stop words list from file
    stopwordsfile = open("./dict/stopwords.txt", "r")
    for word in stopwordsfile: # a stop word in each line
        word = word.replace("\n", '')
        word = word.replace("\r\n", '')
        STOP_WORDS_SET.add(word)
    train_paths = train_path.split(',')
    test_paths = test_path.split(',')
#    train_paths = ['./data/20_newsgroups/comp.sys.ibm.pc.hardware_train/', './data/20_newsgroups/sci.med_train/', './data/20_newsgroups/talk.politics.mideast_train/']
#    test_paths = ['./data/20_newsgroups/comp.sys.ibm.pc.hardware_test/', './data/20_newsgroups/sci.med_test/', './data/20_newsgroups/talk.politics.midast_test/']

    # iterate over the files in the directory.
    train_docs=[]
    trainnum=0
    for document_path in train_paths:
        for document_file in glob.glob(os.path.join(document_path.strip(), '*')):
            document = NewsGroupDocument(document_file)
            document.split(STOP_WORDS_SET) # tokenize
            train_docs.append(document)
            trainnum+=1

    test_docs=[]
    testnum=0
    for document_path in test_paths:
        for document_file in glob.glob(os.path.join(document_path.strip(), '*')):
            document = NewsGroupDocument(document_file)
            document.split(STOP_WORDS_SET) # tokenize
            test_docs.append(document)
            testnum+=1

    print "Number of trainning documents:" + str(trainnum)
    print "Number of testing documents:" + str(testnum)
    return train_docs,test_docs

if __name__ == '__main__':
    pass

