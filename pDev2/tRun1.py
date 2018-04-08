# load the mRun and execute it with different models ... 
# save the results and give a estimation comparing all the model resutls'
# New version 1. 
from datetime import datetime
import mRun as mr
import utils_data as md
import numpy as np
import pandas as pd

rec_eval = True
rec_test = True
tsd       = pd.DataFrame()
tse       = pd.DataFrame()

def get_models(type):
    if   type == "FRFLO":
        return [
            { 'dt':'C2',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 5000, "pe": [], "pt": []  },
            { 'dt':'C4',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 5000, "pe": [], "pt": []  },
            { 'dt':'C1',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 5000, "pe": [], "pt": []  },
        ]
    elif type == "FRALL1":
        return [
            { 'dt':'C2',  "e":20, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
            { 'dt':'C4',  "e":50, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
            { 'dt':'C1',  "e":50, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
            # { 'dt':'C0',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
        ]
    elif type == "FLALL":
            return [
            { 'dt':'C2',  "e":100, "lr":0.0001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
            { 'dt':'C4',  "e":100, "lr":0.0001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
            { 'dt':'C1',  "e":100, "lr":0.0001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
            # { 'dt':'C0',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
        ]
    else: return []

def print_results(execc, typ = "pt"): 
    print("0<60_1>60__0<23_1<60_2<93_3>93  ")
    #open file 
    file = md.LOGDAT +  "/_logs_tr2.txt" # + md.DESC +
    f = open(file, 'w')
    # for i in range(len(md.dsp)): 
    for i in range(10): #40 
        # algorithm to check the probs of the pred.        # check concitions - only when diff > 3 
        gt3 = 0;  gt3, gtM = md.comp_perf(md.dsp.iloc[i,1], execc[2]["pt"][1][i][0]  )
        line = print_line(execc, i, typ)        # if gt3 == 1:  
        print(line )
        
        f.write(line + "\t " +  str(gt3)  + "\n") 
    f.close()       #close file
 
def print_line(execc, i, typ = "pt"): 
    promp = "m:{0:10} \tR-{1:5}   ||" .format(md.dsp.iloc[i,0], md.dsp.iloc[i,1])  
    promp = promp + str([ print_pred(execc[x], typ, i) for x in range(len(execc)) ]  )
    return(promp)

def print_pred( ex , typ, i  ): 
    promp =  "____" + ex["dt"] 
    # promp =  "\t" + ex["dt"]                                       prob.
    if ex["dt"] == "C1": return promp + "{0} ({1})".format(ex[typ][1][i],ex[typ][0][i] ) # + 
    else: return promp + "{0:2}".format(ex[typ][1][i][0])

def download_pandas( ):
    if rec_test: tsd.to_csv(md.LOGDAT + "ttestDS.csv"); print("\nDownload pandas test")
    if rec_eval: tse.to_csv(md.LOGDAT + "tevalDS.csv"); print("\nDownload pandas eval")
    return

def record_data( ex, dsp, dspo, type = "pt"  ):
    # print(len(np.array([str(xi[0]) for xi in ex["pt"][1]])));     print(len(tsd.columns))
    # tsd.insert(len(tsd.columns), ex["dt"] + 'FP_P', dsp2["FP_P"].map(lambda x: dc( x ) )    )
    #tsd[ "_PRED"] = np.array([str(xi[0]) for xi in ex["pt"][1]])  # pandas warnings! 
    dsp.insert(len(dsp.columns), ex["dt"] + '_FP_P',  dspo["FP_P"].map(lambda x: md.dc( x ) ))    
    dsp.insert(len(dsp.columns), ex["dt"] + '_PRDU',  np.array([int(xi[0])   for xi in ex[type][1]])  )    
    dsp.insert(len(dsp.columns), ex["dt"] + '_PRED',  np.array([str(xi)      for xi in ex[type][1]])  )    
    dsp.insert(len(dsp.columns), ex["dt"] + '_PROB',  np.array([str(xi)      for xi in ex[type][0]])  )    
    # error calc: 
    if ex["dt"] == "C1":
        dsp.insert(len(dsp.columns), ex["dt"] + '_ERR', 
                #( dsp.loc[:, ex["dt"] + '_FP_P'] == dsp.loc[:,ex["dt"] +'_PRDU']  )   )  
                np.array([ abs(dsp.loc[i,ex["dt"] +'_PRDU']-dsp.loc[i,ex["dt"] +'_FP_P'])>3 for i in range(len(dsp))  ])  )   
    else:
        dsp.insert(len(dsp.columns), ex["dt"] + '_ERR', ( dsp.loc[:, ex["dt"] + '_FP_P'] == dsp.loc[:,ex["dt"] +'_PRDU']  )  )   
    return dsp

def mainRun(): 
    global tsd, tse
    print("___Start!___" +  datetime.now().strftime('%H:%M:%S')  )
    
    md.DESC = "FRALL1";        # FRFLO   FRALL1    FLALL
    md.spn  = 10000            # 10000
    
    md.setDESC(md.DESC)
    ALL_DS = md.LOGDAT + md.DESC + md.DSC 
    
    # DATA READ  ------------------------------------------------ 
    # md.mainRead2(ALL_DS, 1, 2 ) # , all = True, shuffle = True  ) 
    md.mainRead2(path=ALL_DS, part=1, batch_size=2 ) # For testing I am forced to used JSON - column names and order may be different! 

    url_test = md.LOGDAT + "FREXP1/" ; # url_test = "url"
    force = False; excel = True  # dataFile = "frall2_json.txt"; labelFile = "datal.csv"     
    md.get_tests(url_test, force, excel )

    mr.final = "_" #_ _101
    # OPERATIONS  ------------------------------------------------ 
    md.get_columns(force=force, pp_excel=True)
    if rec_test: tsd = md.dsp[["M", "FP"]]
    if rec_eval: tse = md.dst.iloc[:md.spn, :2]

    execc = get_models(md.DESC)
    for ex in execc:
        # md.spn = ex["spn"]; 
        md.dType = ex["dt"]; mr.epochs = ex["e"]; mr.lr = ex["lr"]; mr.h = ex["h"]
        
        md.normalize()  
        mr.ninp, mr.nout, mr.top_k = md.getnn()
        md.MODEL = mr.get_hpar(mr.epochs, final=mr.final)
        md.MODEL_DIR = md.LOGDIR + md.DESC + '/' +  md.MODEL #+"/" 
        mr.model_path = md.MODEL_DIR + "/model.ckpt" 
        
        mr.build_network3()                                                                                                                                                                                                                                                                                    
        print(mr.model_path) 
        mr.clean_traina()

        # mr.train(it= ex["e"], disp=True, batch_size = 128, compt = True)
        ex["pe"] = mr.evaluate( )
        mr.vis_chart( ) # visualize the training chart
        
        ex["pt"] = mr.tests(url_test, p_col=False  )
        if rec_test: tsd = record_data( ex, tsd, md.dsp, type = "pt")    
        if rec_eval: tse = record_data( ex, tse, md.dst.iloc[:md.spn, :3], type = "pe")    

    # PRINTING  ------------------------------------------------ 
    print("end!___" +  datetime.now().strftime('%H:%M:%S')  )
    print_results(execc, typ = "pt") 
    # DOWNLOAD ------------------------------------------------- 
    download_pandas( )

if __name__ == '__main__':
    mainRun()


