#!/usr/local/bin/python
# coding: gbk
#lib import
import sys
import time
import traceback
from plsa import *
import numpy as np
#your's import
#sys.path.append('./bin')
#from log import * 

#author: Wayne Lin
#input:
#ouput:
#
#
#

def main():
    m = np.array(
        [
            [0.1,0.2,0.3,0.4],
            [0.5,0.6,0.7,0.8],
            [0.3,0.2,0.2,0.1],
        ])
    print m
    out=normalize(m,1)
    print 'by ->'
    print out
    out=normalize(m,0)
    print 'by |v'
    print out
#traceback.print_exc()    

if __name__ == '__main__':
    t1=time.time() 
    main()
    t2=time.time()
    timestr='use time:\t'+str(t2-t1)

