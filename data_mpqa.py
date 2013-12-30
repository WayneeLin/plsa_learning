# coding: gbk
import os
import glob
import sys
from doc import *
from conf import *

STOP_WORDS_SET = set()

#author: Wayne Lin
#deal with input [data/database.mpqa.2.0]

class MpqaDocument(Document):
    def __init__(self, filepath,label):
        Document.__init__(self, filepath)
        #use input topic file's label
        self.label = label 

    def split(self, STOP_WORDS_SET):
        Document.split(self, STOP_WORDS_SET)

def getData(train_path,test_path):    
    # load stop words list from file
    stopwordsfile = open("./dict/stopwords.txt", "r")
    for word in stopwordsfile: # a stop word in each line
        word = word.replace("\n", '')
        word = word.replace("\r\n", '')
        STOP_WORDS_SET.add(word)

    path_head=train_path[:train_path.rfind('/')]
    train_docs=[]
    inf=open(train_path)
    for line in inf:
        label,filen=line.strip().split()
        filenn=path_head+'/'+filen
        document = MpqaDocument(filenn,label)
        
        document.split(STOP_WORDS_SET) # tokenize
        train_docs.append(document)
    inf.close()
    test_docs=[]
    path_head=test_path[:test_path.rfind('/')]
    inf=open(test_path)
    for line in inf:
        label,filen=line.strip().split()
        filenn=path_head+'/'+filen
        document = MpqaDocument(filenn,label)
        document.split(STOP_WORDS_SET) # tokenize
        test_docs.append(document)
    inf.close()
    print "number of trainning documents:" + str(len(train_docs))
    print "Number of testing documents:" + str(len(test_docs))  
    return train_docs,test_docs

if __name__ == '__main__':
    pass

