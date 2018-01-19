# tensorboard --logdir=.\my_graph
# tensorboard => http://localhost:6006 
# jupyter => http://localhost:8889

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

DESC       = "FRFLO" ; pp_abs = False
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
    global COL_DS, ALL_DS, ALL_DSJ, DESC
    DESC = pDESC
    COL_DS     = LOGDAT + DESC + DC 
    ALL_DSJ    = LOGDAT + DESC + DSJ 
    ALL_DS     = LOGDAT + DESC + DSC 

def des(): return DESC+'_'+dType+"_filt:"+  filter[0]+str(filter[1])
def c2(df, rv=1):
    if rv == 1:
        if( df < 60 ):                  return [1,0]  
        elif( df >= 60 ):               return [0,1]      
    elif rv==2: 
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
    lz = [0] * nout
    i = df #//nRange
    # print('{} and {}', (df,dfIndex))
    
    if    0 < i < nout:   lz[i]  = 1 
    elif  i < 0:          lz[0]  = 1  
    elif  i >= nout:      lz[-1] = 1
    
    # if i  >nout: i = nout-1
    # elif i < 0: i = 0 
    # for j in range(i-1, i+1): cN1(i,df)    

    return lz 

def cN1(i, df): 
    val = 1 if i == 1 else 0.5
    if    0 < df < nout:   lz[i]  = 1  # lz[i]  = 1 
    elif  df < 0:          lz[0]  = 1  
    elif  df >= nout:      lz[-1] = 1
    

# Maybe I can do this with hot-encoder in sckitlearn
def cc(x, rv=1):
    global nout
    if   dType == 'C4':  nout = 4;   return c4(x, rv);
    elif dType == 'C1':  nout = 102; return cN(x); 
    elif dType == 'C2':  nout = 2;   return c2(x, rv);
    elif dType == 'C0':  nout = 1;   return [x];
def dc(df, val = 1 ): 
    
    if dType == "C0": return df 
    else: 
        try:    val = df.index(val)
        except: val = 0    
    return val

def read_data2(path): #NOT USED
    pass
    # columns = pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python" ,skiprows=0, nrows=1)
    # columns = columns.columns
    # dst = pd.read_csv( tf.gfile.Open(path), sep=None, skipinitialspace=True,  engine="python" , skiprows=128, nrows=128)
    # # dst = pd.read_csv( tf.gfile.Open(data_path), sep=None, skipinitialspace=True,  engine="python" )
    # dst = dst.fillna(0)
    # dst.insert(2, 'FP_P', dst['FP'].map(lambda x: cc(x)))  
    
def normalize(opt = 1):     
    if opt == 0 or opt == 1: dst['FP_P'] = dst['FP'].map(lambda x: cc( x ))
    if (not flag_dsp) or opt == 0 or opt == 2 : dsp['FP_P'] = dsp['FP'].map(lambda x: cc( x ))
    if (not flag_dsc) or opt == 0 or opt == 3: dsc['FP_P'] = dsc['FP'].map(lambda x: cc( x ))

def read_data1(data_path,  typeSep = True, filt = "", filtn = 0, pand=True, shuffle = True): 
    global dataT; global dataE;
    #read excel by batchs
    dst = pd.read_csv( tf.gfile.Open(data_path), sep=None, skipinitialspace=True,  engine="python" )
    dst = dst.fillna(0)
    if filt == '>':
            dst = dst[dst["FP"]>filtn]
    elif filt == '<':
        dst = dst[dst["FP"]<filtn]

    dst.insert(2, 'FP_P', dst['FP'].map(lambda x: cc(x)))  
    if typeSep == True:
        dataCol = 4 # M F T Cx
        dst_tmp = [rows for _, rows in dst.groupby('Type')]
        del dst
        #dst.loc[:,'FP_P']//[:, dataCol:]  # .as_matrix().tolist()
        dataE   =  {'label' : dst_tmp[0].loc[:,'FP_P'], 'data' : dst_tmp[0].iloc[:,dataCol:]  }
        dataT   =  {'label' : dst_tmp[1].loc[:,'FP_P'], 'data' : dst_tmp[1].iloc[:,dataCol:]  }
    else :  
        dataCol = 3 # M F Cx  
        if shuffle: dst = dst.sample(frac=1).reset_index(drop=True) 
        dataT  = {'label' : dst.loc[spn:,'FP_P'] , 'data' :  dst.iloc[spn:, dataCol:] }
        dataE  = {'label' : dst.loc[:spn-1,'FP_P'] , 'data' :  dst.iloc[:spn, dataCol:] }
        # dataA   =  {'label' : dst.loc[:,'FP_P'], 'data' : dst.loc[:,dataCol:]  }
        
def convert_2List(dst): return {'label' : dst["label"].as_matrix().tolist(), 'data' : dst["data"].as_matrix().tolist()}


def get_batches(batch_size):
    n_batches = int(len( dst.loc[spn:]  ) // batch_size)
    print(n_batches*batch_size)
    # x,y = dataT["data"][:n_batches*batch_size], dataT["label"][:n_batches*batch_size]
    for ii in range(0, len( dst.loc[spn:spn+n_batches*batch_size]) , batch_size ):
        #convert to list! 
        yield dst.iloc[spn+ii: spn+ii+batch_size, 3:].as_matrix().tolist(), dst.loc[spn+ii: spn+ii+batch_size-1, 'FP_P' ].as_matrix().tolist() 

def mainRead(filt=["", 0]):             
    global ninp, nout, dataT, dataE; 
    print(des())
    
    start = time.time()
    
    read_data1(ALL_DS, typeSep = type_sep, filt=filt[0], filtn=filt[1] ) 
    
    elapsed_time = float(time.time() - start)
    print("data read - lenTrain={}-{} & lenEv={}-{} time:{}" .format(len(dataT["data"]), len(dataT["label"]),len(dataE["data"]),len(dataE["label"]),elapsed_time ))

    ninp = len(dataE["data"].columns)
    print("N of columns: {}" .format( str(ninp) ) )
    dataT= convert_2List(dataT)
    dataE= convert_2List(dataE)
    return ninp, nout

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

    # dst.insert(2, 'FP_P', dst['FP'].map(lambda x: cc( x )))  
    # if batch_size > spn: spn = -1
    # dst = dst.sample(frac=1).reset_index(drop=True) 
    # dataT  = {'label' : dst.loc[spn:,'FP_P'] , 'data' :  dst.iloc[spn:, 3:] }
    # dataE  = {'label' : dst.loc[:spn-1,'FP_P'] , 'data' :  dst.iloc[:spn, 3:] }
    # elapsed_time = float(time.time() - start)
    # print("data read - lenTrain={}-{} & lenEv={}-{} time:{}" .format(len(dataT["data"]), len(dataT["label"]),len(dataE["data"]),len(dataE["label"]), elapsed_time ))
    # ninp  = len(dataT["data"].columns)
    # dataT= convert_2List(dataT)
    # dataE= convert_2List(dataE)
    # return ninp, nout
 
def getnn():
    ninp = len(dst.columns) - 3 
    if   dType == 'C4':  nout = 4;   top_k = 3
    elif dType == 'C2':  nout = 2;   top_k = 2
    elif dType == 'C1':  nout = 102; top_k = 5
    elif dType == 'C0':  nout = 1;   top_k = 1
    return ninp, nout, top_k

def check_perf_CN(predv, dataEv, sk_ev=False, sf = True ):
    global dType
    gt3 = 0; gtM = 0; 
    # predvList = predv.tolist()
    # assert(len(predv) == len(dataEv['label']))
    print("denormalization all Evaluation : {} = {}" .format(len(predv[1]), len(dataEv)))
    #for i in range(100):
    for i in range(len(dataEv)):
        if (i % 1000==0): print(str(i)) #, end="__") 
        try:
            # pred_v = dc( predv.tolist()[i], np.max(predv[i]))
            if sf: pred_v = predv[1][i][0]
            else: pred_v = predv[i]
            data_v = dataEv[i] if sk_ev  else dc( dataEv[i])
            gt3, gtM = tuple(map(operator.add, (gt3, gtM), comp_perf(data_v, pred_v, dType )))
        
        except: print("error: i={}, pred={}, data={} -- ".format(i, pred_v, data_v))
    print("Total: {} GT3: {}  GTM: {}".format(len(predv[1]), gt3, gtM)) 
    return gt3, gtM 

def comp_perf( val, pred, dType='C1'): 
    gt3 = 0; gtM = 0;     
    if   dType == 'C4' and pred != val:  gt3=gtM=gtM+1
    elif dType == 'C2' and pred != val:  gt3=gtM=gtM+1
    else: #if dType == 'C1':
        num = abs(pred-val)
        if num > 3: gt3+=1
        if num > 10: gtM+=1
    return gt3, gtM 

def feed_data(dataJJ, d_st = False, pand=False, p_col = False,  p_all = True):
    #index_col=0 if p_abs else 2 #abs=F => 2 == 6D
    index_col = 2 #name - num, abs = cc
    # col_df = pd.read_csv(COL_DS, index_col=index_col, sep=',', usecols=[0,1,2,3])    
    col_df = pd.read_csv(COL_DS, index_col=index_col, sep=',', usecols=[0,1,2,3])    
    col_df = col_df.fillna(0)
    print("input-no={}".format( len(col_df )))
    
    try:
        indx = dst.columns
        indx = indx.delete(2)
    except NameError:  #TypeError
        indx = col_df.index
        indx = indx.insert(0, "M")
        indx = indx.insert(1, "FP")  # indx = indx.insert(2, "FP_P")

    if p_col:    
        dataTest_label = []
        dataJJ = "["
        for i in range(len(col_df)): 
            #fpp = cc( int(  col_df.iloc[i]["fp"]  ))
            fpp = int(  col_df.iloc[i]["fp"]  )
            dataTest_label.append(  fpp ) 
            #dataJJ += '{"m":"'+str(i)+'",'+'"'+str(col_df.iloc[i].name)+'"'+":1},"
            dataJJ += '{"m":"'+str(col_df.iloc[i].name)+'",'+'"'+str(col_df.iloc[i].name)+'"'+":1},"
        dataJJ += '{"m":"0"}]';  dataTest_label.append(0)
        # dataJJ += ']'
        dataJJ = json.loads(dataJJ)

    if pd.core.indexes.numeric.is_integer_dtype(col_df.index):  isInt = True
    else: isInt = False

    json_df  = pd.DataFrame(columns=indx); df_entry = pd.Series(index=indx)
    
    df_entry = df_entry.fillna(0) 
    ccount = Counter()

    if(isinstance(dataJJ, list)):json_data = dataJJ
    else: json_str=open(dataJJ).read();  json_data = json.loads(json_str)
    
    if p_all: ll = range(len(json_data))
    else: ll = range(ll_st, ll_en)

    #for i in range(2):
    for i in ll: # print(i)
        df_entry *= 0
        m = str(json_data[i]["m"])
        df_entry.name = m
        df_entry["M"] = m
        for key in json_data[i]:
            if key != "m": 
                # key_wz = key if pp_abs else int(key)  #str(int(key)) FRFLO - int // FRALL str!
                
                if isInt : key_wz = int(key)  # if comp NOT conatin letters
                else: key_wz = str(key)       # if comp contains letters
                
                try: #filling of key - experimental or COMP 
                    if d_st:
                        ds_comp = col_df.loc[key_wz] #print(ds_comp) # THIS IS THE MOST TIME CONSUMING OP. 
                        col_key = ds_comp.cc if pp_abs else  str(ds_comp.name) #
                    else: 
                        col_key = str(int(key)) 
                    df_entry[col_key] =  np.float32(json_data[i][key]) # df_entry.loc[col_key]
                except: 
                    if d_st: print("m:{}-c:{} not included" .format(m, key_wz)); ccount[key_wz] +=1
        df_entry = df_entry.replace(0, np.nan)
        json_df = json_df.append(df_entry,ignore_index=False)
        if i % 1000 == 0: print("cycle: {}".format(i))
    print("Counter of comp. not included :"); print(ccount) # print(len(ccount))
 
        
    if p_col:  
        json_df["FP"] = dataTest_label
    
    if pand:  return json_df
    else:     
        if p_col: return json_df.iloc[:,3:].as_matrix().tolist(), dataTest_label
        else:     return json_df.iloc[:,3:].as_matrix().tolist() 

# TEST -> ALWAYS WITH 6D ... cx -> conditional 
def get_data_test( desc = 1 ): 
    if desc == 1: 
        json_str = '''[
            { "m":"PBV10476AS", "178583" :0.74598 , "106104" :0.1 , "182789" :0.04 , "130172" :0.035 , "179661" :0.035 , "164421" :0.018 , "600040" :0.0108 , "116165" :0.008 , "164419" :0.0018 , "103396" :0.001 , "130217" :0.001 , "131460" :0.001 , "690750" :0.0007 , "611089" :0.0006 , "130058" :0.0004 , "130354" :0.0002 , "131101" :0.0002 , "131435" :0.00012 , "131136" :0.0001 , "131315" :0.0001 },   
            { "m":"1", "100023" : 1 },
            { "m":"2", "100025" : 1 },
            { "m":"3", "100034" : 1 },
            { "m":"4", "100023" :0.5 , "100034" :0.5 },
            { "m":"10", "100023" :0.5, "100025" :0.5 }] '''
        tmpLab = [50, 73, 75, 46, 60, 75]
    else: 
        json_str =  '''[
            { "m":"8989", "100023" :0.5 },
            { "m":"8988", "100023" :0.5 , "100025" :0.5 }] '''
        tmpLab = [73, 75]
    return json_str, tmpLab

        
def get_tests(url_test='url', force=False, pp_excel=False, pDataFile = "data_jsonX.txt", pLabelFile = "datalX.csv", p_dst=True, p_all = True ): 
    global dsp, flag_dsp 
    
    if flag_dsp or force: 
        flag_dsp = False
        # 2 -- READ EXCEL 
        if pp_excel: dsp = pd.read_csv( tf.gfile.Open( LOGDAT + DESC + "/datasc_tx.csv"  ), sep=None, skipinitialspace=True,  engine="python" )
        else: # 1 -- READ JSON 
            if url_test != 'url':           # test  file 
                json_data = url_test + pDataFile
                tmpLab = pd.read_csv( url_test + pLabelFile, sep=',', usecols=[0,1])    
                tmpLab = tmpLab.loc[:,'fp'].tolist()
                #DESC     = "FREXP1_X"
            else:                           # get data test JSON = url
                json_str, tmpLab = get_data_test(1) 
                json_data = json.loads(json_str)
                #DESC =  'matnrList...'
        
            dsp = feed_data(json_data ,pand=True, d_st=p_dst, p_all = p_all)       #d_st = display status
            
            if p_all: dsp["FP"] = tmpLab      #normalize(2)  # del dsp['FP_P']
        if p_all: dsp.insert(2, 'FP_P', dsp['FP'].map(lambda x: cc( x )))  

    else:    
        return True
      
def get_columns(force=False, pp_excel = False, p_dst=True): 
    global dsc, flag_dsc 
    if flag_dsc or force: 
        if pp_excel : dsc = pd.read_csv( tf.gfile.Open( LOGDAT + DESC + "/datasc_cc.csv"  ), sep=None, skipinitialspace=True,  engine="python" )
        else: 
            flag_dsc = False
            #d_st = display status
            dsc = feed_data(dataJJ="", d_st=p_dst, pand=True, p_col=True) 
            #normalize(3)
            dsc = dsc.drop(dsc.index[-1])
            # del dsc['FP_P']
        dsc.insert(2, 'FP_P', dsc['FP'].map(lambda x: cc( x )))  
    else:    
        return True
    # indx=[];   index_col=0 if p_abs else 2 #abs=F => 2 == 6D
    # col_df = pd.read_csv(COL_DS, index_col=index_col, sep=',', usecols=[0,1,2,3])    
    # col_df = col_df.fillna(0)
    # print("input-no={}".format( len(col_df )))
    # indx = dst.index
        
    # json_df  = pd.DataFrame(columns=dst.columns); df_entry = pd.Series(index=dst.columns)
    # df_entry = df_entry.fillna(0) 
    # for i in range(len(col_df)): 
        # df_entry *= 0
        # df_entry["M"] = i #col_df.iloc[i].name
        # df_entry["FP"] = col_df.iloc[i]["fp"]
        # key = col_df.iloc[i].name
        # key_wz = key if p_abs else (int(key))
        # df_entry[] = 1
        

def testsJ(excel):
    print("tests JSON")    
    dataAll = {'label' : [] , 'data' :  [] }
    json_flag = True    
    if json_flag: 
        json_str = '''[{ "m":"8989", "c1" :0.5, "c3" :0.5  },
                    { "m":"8988", "c3" :0.5 , "c4" :0.5 }] '''
        json_data = json.loads(json_str)  #;print(json_data[0]['m'])
    else: json_data = ALL_DSJ
  
    start = time.time()
    dataAll['data'] = feed_data(json_data, pand=True, d_st=True,  p_exp=False);

    # TO DO: separate between training and evaluation! 
    
    print("data read - time:{}" .format(float(time.time() - start) ))
    down_excel(dataAll['data'], excel)

# ll_st = 0; ll_en=10000; #ll_en=20000; 26000; 32000; 38000; 44000; 
ll_st = 44001; ll_en = 50000; #max = 64829 = HTK10719AU
# ll_st = 0; ll_en = 10;
def testsJ2(excel=True, split = False, pTest = True):
    start = time.time()
    print("___JSON!___" +  datetime.now().strftime('%H:%M:%S')  )

    # url_test = LOGDAT + "FREXP1/" ; dataFile = "data_jsonX.txt";  labelFile = "datalX.csv" ;   #url_test = "url"
    setDESC("FLALL2"); url_test = LOGDAT + "FLALL2/" ; dataFile = "frall2_json.txt"; labelFile = "datal.csv" 
    
    if pTest:                                               #   disp   all
        get_tests(url_test, False, False, dataFile, labelFile, False, False ); tmp = dsp;
    else: 
        get_columns(False, False, False); tmp = dsc
    
    # del tmp['FP_P']

    if split: 
        pass # separate betweent TR and EV     

    print("data read - time:{}" .format(float(time.time() - start) ))
    down_excel(tmp,  excel)

def down_excel(data, excel_flag): 
    if excel_flag: 
        writer = pd.ExcelWriter(LOGDAT+'pandas_datasc.xlsx')
        data.to_excel(writer, sheet_name='Sheet1')
        writer.save()
        print("JSON downloaded into excel! ")

def split_data():
    data_o = "datasc.cvs"
    data_t = "datasct2.cvs" #t100 - keep the order! 

    #shuffle or control split? 

    #control split - I need more information... 

def print_formulation(ds):
    # print(ds.iloc[ds.nonzero()])
    print(*ds.iloc[ds.nonzero()].index) # in 1 line
    print(*ds.iloc[ds.nonzero()])       # in 1 line
    
    print("N. of components: {}".format(len(ds.iloc[ds.nonzero()])  ))




def main1(): 
    print("hi1")
    # md.mainRead2(ALL_DS, 1, 2, all = True, shuffle = True  ) 
    mainRead2(ALL_DS, 1, 2, all = False ) # For testing I am forced to used JSON - column names and order may be different! 
    testsJ2(excel=True, split = False, pTest = True)

def main2():
    url_test = LOGDAT + "FREXP1/" ; # url_test = "url"
    force = False; excel = True  # dataFile = "frall2_json.txt"; labelFile = "datal.csv"     
    get_tests(url_test, force, excel )
    print_formulation( dsp.iloc[2]  )

if __name__ == '__main__':
    main1()


