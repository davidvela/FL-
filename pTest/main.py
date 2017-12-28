# tensorboard --logdir=.\_zfp\data\my_graph
# tensorboard => http://localhost:6006 
# jupyter => http://localhost:8889

import pandas as pd
import tensorflow as tf
import numpy as np
import requests
import json
import time
# import utils_data as md

import sys, os

sys.path.insert(0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pDev import utils_data

del sys.path[0]

from types import *
from collections import Counter
from datetime import datetime

# 2 modes - train all the models - analyse - predict and compare results 
# .
# Load data Options: 
#   - all at once - 1 file, shuffle, separate 
#   - different files ... created in another program! NO - I need eval and train at the same time 
#        disadv - train and test always differ... logic to separate...
#   - one file and read sequentially  
# 
# Train the models all at once will be pretty hard to control ... 
# what I need is to train the models independently and then test them all at once ... 
#   read them independently *** LOOP 
#   read them all at once ... 
# .
#  





def data_constants():
    LOG        = "../../LOG.txt"
    LOGDIR     = "../../data/my_graph/"
    LOGDAT     = "../../data/"

    # spn        = 5000  #5000 -1 = all for training 
    spn = 1
    # DESC       = "ZTEST"
    DESC       = "FRFLO"
    # DESC       = "FRALL1"
    dType      = "C1" #C1 or C4
    MMF        = "MODJJ1" #2(1) OR 5 (4)

    #---------------------------------------------------------------------
    MODEL_DIR  = LOGDIR + DESC + '/' + DESC + dType +  MMF +"/"  
    model_path = MODEL_DIR + "model.ckpt" 
    DSJ        = "/data_json.txt"
    DSC        = "/datasc.csv"   
    DC         = "/datac.csv"
    DL         = "/datal.csv"
    LAB_DS     = LOGDAT + DESC + DL #"../../_zfp/data/FRFLO/datal.csv"
    COL_DS     = LOGDAT + DESC + DC 
    ALL_DSJ    = LOGDAT + DESC + DSJ 
    ALL_DS     = LOGDAT + DESC + DSC 


def mainRun(): 
    # print(get_hpar() ); return 
    # epochs     = 10
    # train(epochs, disp, batch_size)
    # evaluate( )

    
    url_test = "../../_zfp/data/FREXP1/" ;
    # tests(url_test, p_col=False  )
    print("___The end!")

if __name__ == '__main__':
    mainRun()