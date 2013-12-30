# -*- coding: gbk -*-
import numpy as np
import heapq
import os,sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)


"""
Implementation of probabilistic Latent Semantic Analysis/Indexing as described
in
"Probabilistic Latent Semantic Indexing", Hofmann, SIGIR99

Notation:
    w: word
    d: document
    z: topic

    V: vocabulary size
    D: number of documents
    Z: number of topics

    p_z:        p(z)   Z
    p_w_z:      p(w|z) V*Z or W*Z
    p_d_z:      p(d|z) D*Z

    p_z_d_w:    p(z|d,w) Z*D*V
                p_z_d_w= p_z_old * p_d_z_old[d,:] * p_w_z_old[w,:]

"""


'''
#_plsa is plsa in c
try:
    import _plsa
    HAVE_EXT = True
except:
    HAVE_EXT = False
'''
#normalize a vector
def normalize_1d(a, out=None):
    if out is None: out = np.empty_like(a)
    s =float( np.sum(a))
    if s != 0.0 and len(a) != 1:
        np.divide(a, s, out)
    return out

#normalize a matrix or a vector
def normalize(M, axis=0, out=None):
    #normalize a vector
    if len(M.shape) == 1: return normalize_1d(M, out)
    #normalize a matrix
    if out is None: out = np.empty_like(M)

    #if axis !=0,normalize matrix by vertical axis |
    # normalize by horizontal axis default --
    if axis == 0:
        M = M.T #M.swapaxes(0,1)
        out = out.T

    for i in range(len(M)):
        normalize_1d(M[i], out[i])
    if axis == 0: out = out.T
    return out

def loglikelihood(td, p_z, p_w_z, p_d_z):
    """
    Compute the log-likelihood that the model generated the data.
    """
    V, D = td.shape
    L = 0.0
    for w,d in zip(*td.nonzero()):
        p_d_w = np.sum(p_z * p_w_z[w,:] * p_d_z[d,:])
        if p_d_w > 0: L += td[w,d] * np.log(p_d_w)
    return L

def train(td,
          p_z, p_w_z, p_d_z,
          p_z_old, p_w_z_old, p_d_z_old,
          maxiter, eps,
          folding_in, debug):

    R = td.sum() # total number of word counts
    lik = loglikelihood(td, p_z, p_w_z, p_d_z)
    for iteration in range(1, maxiter+1):

        # Swap old and new
        p_d_z_old, p_d_z = (p_d_z, p_d_z_old)
        p_w_z_old, p_w_z = (p_w_z, p_w_z_old)
        p_z_old, p_z = (p_z, p_z_old)

        # Set to 0.0 without memory allocation
        p_d_z *= 0.0
        if not folding_in:
            p_w_z *= 0.0
            p_z *= 0.0
        
        for w,d in zip(*td.nonzero()):
            
            # E-step
            p_z_d_w = p_z_old * p_d_z_old[d,:] * p_w_z_old[w,:]
            normalize(p_z_d_w, out=p_z_d_w)
            # M-step
            s = td[w,d] *  p_z_d_w
            p_d_z[d,:] += s

            if not folding_in:
                p_w_z[w,:] += s
                p_z += s

        if folding_in:
            pass
#            print 'p_z_d_w',p_z_d_w

        # normalize
        normalize(p_d_z, axis=0, out=p_d_z)

        if not folding_in:
            normalize(p_w_z, axis=0, out=p_w_z)
            p_z /= R
        lik_new = loglikelihood(td, p_z, p_w_z, p_d_z)
        lik_diff = lik_new - lik
        lik = lik_new
        if debug:
            print "Iteration:", iteration
            print "Parameter change"
            print "P(z): ", np.abs(p_z - p_z_old).sum()
            print "P(w|z): ", np.abs(p_w_z - p_w_z_old).sum()
            print "P(d|z): ", np.abs(p_d_z - p_d_z_old).sum()
            print 'likelihood:#%s,likelihood diff:%s'%(str(lik_new),str(lik_diff))
            if folding_in:
                pass
#                print "!!P(d|z): ", p_d_z

        try:
            assert(lik_diff >= -1e-10)
        except:
            print "lik_diff >= -1e-10, stopping EM at iteration", iteration
            break
        if lik_diff < eps:
            print "No more progress, stopping EM at iteration", iteration
            break

class pLSA(object):

    def __init__(self, model=None,debug=False):
        """
        model: a model, as returned by get_model() or train().
        """
        self.p_z = None
        self.p_w_z = None
        self.p_d_z = None
        if model is not None: self.set_model(model)
        self.debug = debug

    def random_init(self, Z, V, D,folding_in=False):
        """
        Z: the number of topics desired.
        V: vocabulary size.
        D: number of documents.
        """
        # np.random.seed(0) # uncomment for deterministic init
        if self.p_z is None:
            self.p_z = normalize(np.random.ranf(size=Z))
        if self.p_w_z is None:
            self.p_w_z = normalize(np.random.ranf(size=(V,Z)), axis=0)
        if self.p_d_z is None:
            self.p_d_z = normalize(np.random.ranf(size=(D,Z)), axis=0)

    def train(self, td, Z, maxiter=500, eps=0.01, folding_in=False):
        """
        Train the model.

        td: a V x D term-document matrix of term-counts :term_doc_count
        Z: number of topics desired.

        td can be dense or sparse (dok_matrix recommended).
        """
        V, D = td.shape
        self.random_init(Z, V, D,folding_in)
        #if the task is test, don't need to update p_w_z and p_z
        if not folding_in:
            p_w_z_old = np.zeros_like(self.p_w_z)
            p_z_old = np.zeros_like(self.p_z)
        else:
            p_w_z_old=self.p_w_z
            p_z_old=self.p_z
        p_d_z_old = np.zeros_like(self.p_d_z)
        
#       call cpython
#        train_func = _plsa.train if HAVE_EXT else train
        train_func = train

        train_func(td.astype(np.uint32),
                   self.p_z, self.p_w_z, self.p_d_z,
                   p_z_old, p_w_z_old, p_d_z_old,
                   maxiter, eps,
                   folding_in, self.debug)

        return self.get_model()

    def document_topics(self):
        """
        Compute the probabilities of documents belonging to topics.
        Return: a Z x D matrix of P(z|d) probabilities.
        Note: This can be seen as a dimensionality reduction since a Z x D
        matrix is obtained from a V x D matrix, where Z << V.
        """
        return normalize((self.p_d_z * self.p_z[np.newaxis,:]).T, axis=0)

    def document_cluster(self):
        """
        Find the main topic (cluster) of documents.
        Return: a D-array of cluster indices.
        """
        all_doctopic=self.document_topics()
        return all_doctopic.argmax(axis=0)

    def document_topNtopic(self,N):
        """
        Find the TopN topic (cluster) of documents.
        Return: a N*D-array of topic indexs,a N*D-array of topic value.
        """
        all_doctopic=self.document_topics()
        #argsort in accending order!!
        topN_index=np.argsort(all_doctopic,axis=0)[-N:,:]

        x,y=topN_index.shape
        topN_p=np.empty((x,y),dtype=float)
        for i in range(x):
            for j in range(y):
                topN_p[i,j]=all_doctopic[topN_index[i,j],j]
        
        return topN_index,topN_p

    def word_topics(self):
        """
        Compute the probabilities of words belonging to topics.
        Return: a Z x V matrix of P(z|w) probabilities.
        """
        return normalize((self.p_w_z * self.p_z[np.newaxis,:]).T, axis=0)

    def word_cluster(self):
        """
        Find the main topic (cluster) of words.
        Return: a D-array of cluster indices.
        """
        return self.word_topics().argmax(axis=0)

    def topic_labels(self, inv_vocab, N=10):
        """
        For each topic z, find the N words w with highest probability P(w|z).
        inv_vocab: a term-index => term-string dictionary
        Return: Z lists of N words.
        """
        Z = len(self.p_z)
        ret = []
        for z in range(Z):
            ind = np.argsort(self.p_w_z[:,z])[-N:][::-1]
            ret.append([inv_vocab[i] for i in ind])
        return ret

    def get_model(self):
#       return (self.p_z, self.p_w_z, self.p_d_z)
        return (self.p_z, self.p_w_z)

    def set_model(self, model):
#       self.p_z, self.p_w_z, self.p_d_z = model
        self.p_z, self.p_w_z = model
        self.p_d_z =None

