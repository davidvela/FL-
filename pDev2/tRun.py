# load the mRun and execute it with different models ... 
# save the results and give a estimation comparing all the model resutls'
from datetime import datetime
import mRun as mr
import utils_data as md

def get_models(type):
    if type == "FRFLO":
        return [
            { 'dt':'C2',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 5000, "pe": [], "pt": []  },
            { 'dt':'C4',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 5000, "pe": [], "pt": []  },
            { 'dt':'C1',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 5000, "pe": [], "pt": []  },
        ]
    elif type == "FRALL1":
        return [
            { 'dt':'C2',  "e":40,  "lr":0.001, "h":[100 , 100], "spn": 40000, "pe": [], "pt": []  },
            # { 'dt':'C4',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 40000, "pe": [], "pt": []  },
            # { 'dt':'C1',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 40000, "pe": [], "pt": []  },
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

def mainRun(): 
    print("___Start!___" +  datetime.now().strftime('%H:%M:%S')  )
    final = "_" ;  md.DESC = "FRALL1";  # FRFLO   FRALL1
    ALL_DS = md.LOGDAT + md.DESC + md.DSC 
    
    execc = get_models(md.DESC)

    # DATA READ  ------------------------------------------------ 
    # md.mainRead2(ALL_DS, 1, 2 ) # , all = True, shuffle = True  ) 
    md.mainRead2(path=ALL_DS, part=1, batch_size=2 ) # For testing I am forced to used JSON - column names and order may be different! 

    url_test = md.LOGDAT + "FREXP1/" ; # url_test = "url"
    force = False; excel = True  # dataFile = "frall2_json.txt"; labelFile = "datal.csv"     
    md.get_tests(url_test, force, excel )

    # OPERATIONS  ------------------------------------------------ 
    # md.get_columns(force)
    for ex in execc:
        md.spn = ex["spn"]; md.dType = ex["dt"]; mr.epochs = ex["e"]; mr.lr = ex["lr"]; mr.h = ex["h"]
        
        md.normalize()  
        mr.ninp, mr.nout, mr.top_k = md.getnn()
        md.MODEL_DIR = md.LOGDIR + md.DESC + '/'   + mr.get_hpar(mr.epochs, final=final) +"/" 
        mr.model_path = md.MODEL_DIR + "model.ckpt" 
        mr.build_network3()                                                                                                                                                                                                                                                                                    
        print(mr.model_path)    
        # ex["pe"] = mr.evaluate( )
        ex["pt"] = mr.tests(url_test, p_col=False  )

    # PRINTING  ------------------------------------------------ 
    print("end!___" +  datetime.now().strftime('%H:%M:%S')  )
    # print_results(execc, typ = "pt") 
    
    # DOWNLOAD ------------------------------------------------- 
    download_pandas(execc)

def download_pandas(execc):
    # DOWNLOAD EXCEL! ------------------------------------------------ 
    # create a pandas and create new columns - like     
    print("\nDownload pandas")
    # return "hola"
    
    # 2 datasets: 
    # evd = md.dst
    for ex in execc:
        tsd = md.dsp[["M", "FP"]]
        # tsd["PRED"] = ex["pe"][1][i][0] # pred 1
        tsd[ex["dt"] + "_PRED"] = ex["pe"][1] # pred 1
        tsd[ex["dt"] + "_PROB"] = ex["pe"][0] # prob 
        tsd.to_csv(md.LOGDAT + "testDS.csv")

if __name__ == '__main__':
    mainRun()


