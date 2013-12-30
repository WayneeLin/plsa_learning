import sys
import re
import gc
import numpy as np

"""
Author: 
Alex Kong (https://github.com/hitalex)

Reference:
http://blog.tomtung.com/2011/10/plsa
"""

np.set_printoptions(threshold='nan')

class Document(object):

    '''
    Splits a text file into an ordered list of words.
    '''

    # List of punctuation characters to scrub. Omits, the single apostrophe,
    # which is handled separately so as to retain contractions.
    PUNCTUATION = ['(', ')', ':', ';', ',', '-', '!', '.', '?', '/', '"', '*']

    # Carriage return strings, on *nix and windows.
    CARRIAGE_RETURNS = ['\n', '\r\n']

    # Final sanity-check regex to run on words before they get
    # pushed onto the core words list.
    WORD_REGEX = "^[a-z']+$"


    def __init__(self, filepath):
        '''
        Set source file location, build contractions list, and initialize empty
        lists for lines and words.
        '''
        self.filepath = filepath
        self.file = open(self.filepath)
        self.lines = []
        self.words = []

    def split(self, STOP_WORDS_SET):
        '''
        Split file into an ordered list of words. Scrub out punctuation;
        lowercase everything; preserve contractions; disallow strings that
        include non-letters.
        '''
        for line in self.file:
            line=line.strip()
            if line=='':
                continue
            self.lines.append(line)
        for line in self.lines:
            words = line.split(' ')
            for word in words:
                clean_word = self._clean_word(word)
                if clean_word and (clean_word not in STOP_WORDS_SET) and (len(clean_word) > 1): # omit stop words
                    self.words.append(clean_word)
        self.file.close()


    def _clean_word(self, word):
        '''
        Parses a space-delimited string from the text and determines whether or
        not it is a valid word. Scrubs punctuation, retains contraction
        apostrophes. If cleaned word passes final regex, returns the word;
        otherwise, returns None.
        '''
        word = word.lower()
        for punc in Document.PUNCTUATION + Document.CARRIAGE_RETURNS:
            word = word.replace(punc, '').strip("'")
        return word if re.match(Document.WORD_REGEX, word) else None


#use whole word of training data & testing data as the vocabulary
def build_vocabulary(documents1,documents2):
    vocabulary={}
    '''
    Construct a list of unique words in the corpus.
    '''
    # ** ADD ** #
    # exclude words that appear in 90%+ of the documents
    # exclude words that are too (in)frequent
    discrete_set = set()
    for document in documents1:
        for word in document.words:
            discrete_set.add(word)
    for document in documents2:
        for word in document.words:
            discrete_set.add(word)
    vocabulary_list = list(discrete_set)
    for i,v in enumerate(vocabulary_list):
        vocabulary[v]=i
    return vocabulary,vocabulary_list

#get doc in documents[] term count
#   term_doc_count, named "td" in plsa.py
#   a V x D term-document matrix of term-counts
def get_term_doc_count(vocabulary,documents):
    number_of_documents = len(documents)
    vocabulary_size = len(vocabulary)

    tmp_term_doc_count = np.zeros([number_of_documents, vocabulary_size], dtype = np.int)
    term_doc_count = np.zeros([vocabulary_size,number_of_documents], dtype = np.int)
    for d_index, doc in enumerate(documents):
        term_count = np.zeros(vocabulary_size, dtype = np.int)
        for word in doc.words:
            #if the word in testdata not in vocabulary,  ignore this word
            
            if word in vocabulary:
                w_index = vocabulary[word]
                term_count[w_index] = term_count[w_index] + 1
#        del doc.words
#        gc.collect()

        tmp_term_doc_count[d_index] = term_count
    term_doc_count=np.transpose(tmp_term_doc_count) 
    return term_doc_count


