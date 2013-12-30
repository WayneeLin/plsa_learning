# coding: gbk
import os
import glob
import sys
import gc
from doc import *

STOP_WORDS_SET = set()

#author: Wayne Lin
#deal with input [./data/20_newsgroups/]
# using memory: 19g 
classdic={
'C000007':'汽车',
'C000008':'财经',
'C000010':'IT',
'C000013':'健康',
'C000014':'体育',
'C000016':'旅游',
'C000020':'教育',
'C000022':'招聘',
'C000023':'文化',
'C000024':'军事',
}
class ChNewsDocument(Document):

    def __init__(self, filepath,label):
        Document.__init__(self, filepath)
        #use filepath for label
        self.label = label 

    def split(self):
        '''
            Chinese words has been tookenized
        '''
        for line in self.file:
            line=line.strip()
            if line=='':
                continue
            self.lines.append(line)
        for line in self.lines:
            words = line.split(' ')
            for word in words:
                self.words.append(word)
        
        del self.lines
        gc.collect()

        self.file.close()

def getData(train_path,test_path):    
    train_docs=[]
    test_docs=[]

    # iterate over the files in the directory.
    for root, dirs, files in os.walk(train_path):
        for name in files:
            if not str(name).endswith(".txt"):
                continue
            root=str(root)
            fclass1=root[root.rfind('C'):] 
            infn=str(os.path.join(root, name))
            fclass=classdic[fclass1]
            document = ChNewsDocument(infn,fclass)
            document.split()
            train_docs.append(document)

    for root, dirs, files in os.walk(test_path):
        for name in files:
            if not str(name).endswith(".txt"):
                continue
            root=str(root)
            fclass1=root[root.rfind('C'):] 
            infn=str(os.path.join(root, name))
            fclass=classdic[fclass1]
            document = ChNewsDocument(infn,fclass)
            document.split()
            test_docs.append(document)

    print "Number of trainning documents:" + str(len(train_docs))
    print "Number of testing documents:" + str(len(test_docs))
    return train_docs,test_docs

if __name__ == '__main__':
    pass

