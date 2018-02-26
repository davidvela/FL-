import pandas as pd
import tensorflow as tf
import numpy as np

import requests
import json
import sys
import os
import time
import operator
from types import *
from collections import Counter
from datetime import datetime

LOG = "../../LOG.txt"
LOGDIR     = "../../dmodels/"
LOGDAT     = "../../data/"
DSJ        = "/data_json.txt"
DSC        = "/datasc.csv"   
DC         = "/datac.csv"
DL         = "/datal.csv"

#---------------------------------------------------------------------
filter     = ["", 0]
type_sep   = False

DESC       = "FRaFLO" ; pp_abs = False
spn        = 5000  
dType      = "C4" #C1, C2, C4

# DESC       = "FRALL1"; pp_abs = True
# spn        = 10000  #5000 -1 = all for training 

#---------------------------------------------------------------------
MODEL_DIR  = LOGDIR + DESC + '/'   

# LAB_DS     = LOGDAT + DESC + DL # NOT USED!
COL_DS     = LOGDAT + DESC + DC 
ALL_DSJ    = LOGDAT + DESC + DSJ 
ALL_DS     = LOGDAT + DESC + DSC 


nout   = 100
ninp   = 0
dataT  = {'label' : [] , 'data' :  [] } #inmutables array are faster! 
dataE  = {'label' : [] , 'data' :  [] }

flag_dsp = True        
flag_dsc = True 

def setDESC(pDESC): 
    global COL_DS, ALL_DS, ALL_DSJ, DESC, pp_abs
    print("set desc!")
    DESC = pDESC
    COL_DS     = LOGDAT + DESC + DC 
    ALL_DSJ    = LOGDAT + DESC + DSJ 
    ALL_DS     = LOGDAT + DESC + DSC 
    pp_abs     = False
    # pp_abs = False if pDESC == "FRFLO" else True

def mainRead2(path, part, batch_size,  all = True, shuffle = True):  # read by partitions! 
    global spn, dst #ninp, nout, dataT, dataE, spn;
    start = time.time()
    if all:  dst = pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python" )
    else:     
        columns = pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python" ,skiprows=0, nrows=1)
        dst = pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python" ,skiprows=part*batch_size+1, 
                           nrows=batch_size, names = columns.columns)
    
    dst = dst.fillna(0)
    
    if shuffle: dst = dst.sample(frac=1).reset_index(drop=True) 
    dst.insert(2, 'FP_P', dst['FP'] )  
    
    elapsed_time = float(time.time() - start)
    print("data read - {} - time:{}" .format(len(dst), elapsed_time ))


def simplify_ds( dst, path_down):
    pass 


def main1(): 
    print("hi1")
    # md.mainRead2(ALL_DS, 1, 2, all = True, shuffle = True  ) 
    pDesc = "FLALL"
    setDESC(pDesc);
    mainRead2(ALL_DS, 1, 2, all = False ) # For testing I am forced to used JSON - column names and order may be different! 
    # testsJ2(pDesc = pDesc, excel=True, split = False, pTest = False)

def main3(): 
    mainRead2(ALL_DS, 1, 10, all = False, shuffle = True  ) ; normalize()
    dst.

if __name__ == '__main__':
    main3()

