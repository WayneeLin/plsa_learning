# coding: gbk    
import os
import sys
import gc
from plsa import *
#diffenent input
from conf import *

TOPIC_NUM=plsa_cf.getint('PLSA','TOPIC_NUM')
MAX_ITERATIONS=plsa_cf.getint('PLSA','MAX_ITERATIONS')
TOPIC_TOPN_WORDS=plsa_cf.getint('PLSA','TOPIC_TOPN_WORDS')
DOC_TOPN_TOPIC=plsa_cf.getint('PLSA','DOC_TOPN_TOPIC')
WHICH_INPUT=plsa_cf.getint('INPUT','WHICH_INPUT')
TRAIN_PATH=plsa_cf.get('INPUT','TRAIN_PATH')
TEST_PATH=plsa_cf.get('INPUT','TEST_PATH')

OUTPUT_PATH_HEAD=''
if WHICH_INPUT==1:
    from data_mpqa import *
    OUTPUT_PATH_HEAD='mpqa_'
elif WHICH_INPUT==2: 
    from data_newsgroup import *
    OUTPUT_PATH_HEAD='newsgroup_'
elif WHICH_INPUT==3: 
    from data_chnews import *
    OUTPUT_PATH_HEAD='chnews_'
OUTPUT_PATH_HEAD='./output/'+OUTPUT_PATH_HEAD
try:
    assert(DOC_TOPN_TOPIC<=TOPIC_NUM)
except:
    print "DOC_TOPN_TOPIC>TOPIC_NUM err!"
    sys.exit()

def print_doc_res(doc_res,docs,topic_labels,filename):
    outf=open(filename,'w')
    for i in range(len(doc_res)):
    
        outf.write('%s--------------\n'%i)
        topic_index=doc_res[i]
        topic_str='\t'.join(topic_labels[topic_index])
        outf.write('topic:\t%s\t%s\n'%(topic_index,topic_str))
        outf.write('file label:'+docs[i].label+'\n')
        content = '\t'.join(docs[i].words)
        outf.write(content+'\n')
    outf.close()

def print_doc_topNres(doc_res_i,doc_res_p,docs,topic_labels,filename,print_label=False):
    outf=open(filename,'w')
    doc_res_i1=doc_res_i.T
    doc_res_p1=doc_res_p.T
    for i in range(len(doc_res_i1)):
        outf.write('file:%s--------------\n'%i)
        all_labelword=[]
        for j in reversed(range(len(doc_res_i1[i]))):
            topic=doc_res_i1[i][j]
            topics=str(topic)
            topicValue=str(doc_res_p1[i][j])
            if print_label:
                topic_str='\t'.join(topic_labels[topic])
                topic_str='topic#'+topics+':\tprob:'+topicValue+'\t'+topic_str
                all_labelword.append(topic_str)
        if print_label:
            for labelstr in all_labelword:
                outf.write(labelstr+'\n')
        outf.write('file label:'+docs[i].label+'\n')
        content = '\t'.join(docs[i].words)
        outf.write(content+'\n')
    outf.close()

def print_topic_label(topic_labels,filename):
    outf=open(filename,'w')
    for i in range(len(topic_labels)):
        outf.write('Topic#%s\n'%i);
        outstr=''
        for word in topic_labels[i]:
            outstr+=str(word)+' '
        outstr=outstr[:-1]
        outf.write(outstr+'\n')
        outf.write('--------------\n')
    outf.close()

def main():

    train_docs,test_docs=getData(TRAIN_PATH,TEST_PATH)
    
    #use whole word of training data & testing data as the vocabulary
    vocabulary,vocabulary_list=build_vocabulary(train_docs,test_docs)
    train_term_doc_count=get_term_doc_count(vocabulary,train_docs)
    
    test_term_doc_count=get_term_doc_count(vocabulary,test_docs)
    print "Vocabulary size:" + str(len(vocabulary))

#    del train_docs
#    gc.collect()

    plsac = pLSA(debug=True)
    print 'training-------'
    model=plsac.train(
            train_term_doc_count,
            TOPIC_NUM,
            MAX_ITERATIONS)
    doc_toptopic=plsac.document_cluster()
    doc_topNtopic_i,doc_topNtopic_p=plsac.document_topNtopic(DOC_TOPN_TOPIC)
    train_word_res=plsac.word_cluster()
    train_topic_label=plsac.topic_labels(vocabulary_list,TOPIC_TOPN_WORDS)

    print_topic_label(train_topic_label,OUTPUT_PATH_HEAD+'topic_labels.txt')
    print_doc_res(doc_toptopic,train_docs,train_topic_label,OUTPUT_PATH_HEAD+'train_doctopic_result.txt')
    print_doc_topNres(doc_topNtopic_i,doc_topNtopic_p,train_docs,train_topic_label,OUTPUT_PATH_HEAD+'train_doctopic_topN_result.txt',print_label=True)
    print 'testing-------'
    outf=open(OUTPUT_PATH_HEAD+'test_doctopic_result.txt','w')
    plsac_t = pLSA(model,debug=True)
    model_t=plsac_t.train(
            test_term_doc_count,
            TOPIC_NUM,
            MAX_ITERATIONS,
            folding_in=True
            )
    doc_toptopic_test=plsac_t.document_cluster()
    doc_topNtopic_i_test,doc_topNtopic_p_test=plsac_t.document_topNtopic(DOC_TOPN_TOPIC)

    print_doc_res(doc_toptopic_test,test_docs,train_topic_label,OUTPUT_PATH_HEAD+'test_doctopic_result.txt')
    print_doc_topNres(doc_topNtopic_i_test,doc_topNtopic_p_test,test_docs,train_topic_label,OUTPUT_PATH_HEAD+'test_doctopic_topN_result.txt',print_label=True)
    return
    for index in range(len(test_docs)):
        print index,'----'
        topic_index=plsac_t.folding_in(
            test_term_doc_count[:,index],
            MAX_ITERATIONS
            )
        outf.write('%s--------------\n'%index)
        topic_str='\t'.join(train_topic_label[topic_index])
        outf.write('topic:\t%s\t%s\n'%(topic_index,topic_str))
        outf.write('file label:'+test_docs[index].label+'\n')
        content = '\t'.join(test_docs[index].lines)
        outf.write(content+'\n')
    outf.close()

    print_doc_res(test_doc_res,test_docs,train_topic_label,'test_doctopic_result.txt')
    '''
    test_word_res=plsac.word_cluster()
    print 'test_doc_res----------------------------'
    print test_doc_res
    print 'test_word_res----------------------------'
    print test_word_res
    '''

if __name__ == "__main__":
    main()
