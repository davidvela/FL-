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
            { 'dt':'C2',  "e":40, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
            { 'dt':'C4',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
            { 'dt':'C1',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
            # { 'dt':'C0',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
        ]
    else: return []

def print_results(execc, typ = "pt"): 
    print("0<60_1>60__0<23_1<60_2<93_3>93  ")
    #open file 
    file = md.LOGDAT + md.DESC + "/_logs_tr2.txt"
    f = open(file, 'w')
        
    for i in range(len(md.dsp)): #40
        # print("m:{0:15} - R-{4:5}   ||_____________C1-{1:2}_____________C2-{2:2}_____________C3-{3}" 
        # .format(md.dsp.iloc[i,0], execc[0]["pt"][1][i][0], execc[1]["pt"][1][i][0],  execc[2]["pt"][1][i], md.dsp.iloc[i,1], ))    

        # algorithm to check the probs of the pred.
        # .. 

        # check concitions - only when diff > 3 
        gt3 = 0;  gt3, gtM = md.comp_perf(md.dsp.iloc[i,1], execc[2]["pt"][1][i][1]  )
        # gt3 = 1;
        # if gt3 == 1:  
        line = print_line(execc, i, typ)
        print(line )
        f.write(line + "\t " +  str(gt3)  + "\n")

    
    #close file 
    f.close()

def print_line(execc, i, typ = "pt"): 
    promp = "m:{0:15} \tR-{1:5}   \t||" .format(md.dsp.iloc[i,0], md.dsp.iloc[i,1])  
    promp = promp + str([ print_pred(execc[x], typ, i) for x in range(len(execc)) ]  )
    return(promp)

def print_pred( ex , typ, i  ): 
    promp =  "_____________" + ex["dt"] 
    # promp =  "\t" + ex["dt"] 
    if ex["dt"] == "C1": return promp + "{0}".format(ex[typ][1][i])
    else: return promp + "{0:2}".format(ex[typ][1][i][0])

def mainRun(): 
    print("___Start!___" +  datetime.now().strftime('%H:%M:%S')  )
    final = "_" ;  md.DESC = "FRALL1";  # FRFLO   FRALL1
    ALL_DS = md.LOGDAT + md.DESC + md.DSC 
    
    execc = get_models(md.DESC)

    # DATA READ 
    # md.mainRead2(ALL_DS, 1, 2 ) # , all = True, shuffle = True  ) 
    md.mainRead2(path=ALL_DS, part=1, batch_size=2 ) # For testing I am forced to used JSON - column names and order may be different! 
    

    url_test = md.LOGDAT + "FREXP1/" ; # url_test = "url"
    force = False; excel = True  # dataFile = "frall2_json.txt"; labelFile = "datal.csv"     
    md.get_tests(url_test, force, excel )

    # md.get_columns(force)

    for ex in execc:
        md.spn = ex["spn"]; md.dType = ex["dt"]; mr.epochs = ex["e"]
        
        md.normalize()
        mr.ninp, mr.nout, mr.top_k = md.getnn()
        md.MODEL_DIR = md.LOGDIR + md.DESC + '/'   + mr.get_hpar(mr.epochs, final=final) +"/" 
        mr.model_path = md.MODEL_DIR + "model.ckpt" 
        mr.build_network3()
        print(mr.model_path)    

        # mr.evaluate( )

        ex["pt"] = mr.tests(url_test, p_col=False  )

    print("end!___" +  datetime.now().strftime('%H:%M:%S')  )
    print_results(execc, typ = "pt") 

if __name__ == '__main__':
    mainRun()



