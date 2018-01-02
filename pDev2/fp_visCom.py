import pandas as pd 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import time 


#*********************************************************************
# ENCODE 
#*********************************************************************  
def c2(df, rv=1):
    if rv == 1:
        if( df < 60 ):                  return [1,0]  
        elif( df >= 60 ):               return [0,1]      
    elif rf==2: 
        if( df < 60 ):                  return 0
        elif( df >= 60 ):               return 1
def c4(df, rv=1):
    
    if rv == 1:
        if( df < 23 ):                  return [1,0,0,0]  #0
        elif( df >= 23 and df < 60 ):   return [0,1,0,0]  #1
        elif( df >= 60 and df < 93 ):   return [0,0,1,0]  #2
        elif( df >= 93 ):               return [0,0,0,1]  #3    
    elif rv==2: 
        if( df < 23 ):                  return 0
        elif( df >= 23 and df < 60 ):   return 1
        elif( df >= 60 and df < 93 ):   return 2
        elif( df >= 93 ):               return 3
    # elif rf==3: 
    #     if  ( df == [1,0,0,0] ):        return 0 
    #     elif( df == [0,1,0,0] ):        return 1
    #     elif( df == [0,0,1,0] ):        return 2  
    #     elif( df == [0,0,0,1] ):        return 3  
def cN(df):
    global nout
    listofzeros = [0] * nout
    dfIndex = df #//nRange
    # print('{} and {}', (df,dfIndex))
    if    0 < dfIndex < nout:   listofzeros[dfIndex] = 1
    elif  dfIndex < 0:          listofzeros[0]       = 1
    elif  dfIndex >= nout:      listofzeros[nout-1]  = 1
    
    return listofzeros 

def cc(x, rv=1):
    global nout
    if   dType == 'C4':  nout = 4;   return c4(x, rv);
    elif dType == 'C1':  nout = 102; return cN(x); 
    elif dType == 'C2':  nout = 2;   return c2(x, rv);

#*********************************************************************
# paint 
#*********************************************************************  
def paint_plot(data,  component ,clase,m="o"):
    colors = list()
    palette = {0: "red", 1: "green", 2: "blue"}
    x = data[ data['FPP'] == clase ]["M"]
    y = data[ data['FPP'] == clase ][component]
    plt.scatter(x, y , edgecolors='k',s=50, alpha=0.9, marker=m,label=str(clase))
    
def paint(data, component):
    #color=plt.rainbow(np.linspace(0,1,nn))
    for i in rr: 
        #c=next(color)
        paint_plot(data, component, i)
    plt.show()

start = time.time()
# 1- Get data: 
path = outfile = '../../data/FRFLO/datasc.csv' 
# path = outfile = '../../data/FLALL/datasc.csv' 
dst  =  pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python")
elapsed_time = float(time.time() - start)
print(elapsed_time)
com = "160102" #c922 - 160102 - 121 dipropylene glycol 
com = "131104" #c738 - 131104 - 44  hexenol cis 3 
com = "100023"
# read components file ... 
path = outfile = '../../data/FRFLO/datac.csv' 
# path = outfile = '../../data/FLALL/datasc.csv' 
col_df = pd.read_csv(path, index_col=2, sep=',', usecols=[0,1,2,3])    
# col_df.head()

#print component 
print(col_df.loc[int(com)])

# Type of normalization 
dType = "C4"
if   dType == 'C4':  nn = 4
elif dType == 'C1':  nn = 100
elif dType == 'C2':  nn = 2
rr = range(nn)

# dst.insert(2, 'FPP', dst['FP'].map(lambda x: cc(x)))  
dst["FPP"] = dst['FP'].map(lambda x: cc(x, rv=2))

# drop all the columns except the index, 2 and the com 
dst.columns[2:4]
dstt = dst[["M", "FPP", com]]
dst.head()
dstt = dstt.dropna()
# dstt

paint(dst, com)