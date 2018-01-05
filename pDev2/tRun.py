# load the mRun and execute it with different models ... 
# save the results and give a estimation comparing all the model resutls'
from datetime import datetime
import mRun as mr
import utils_data as md

execc = [
{ 'dt':'C2',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 5000, "pe": [], "pt": []  },
{ 'dt':'C4',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 5000, "pe": [], "pt": []  },
{ 'dt':'C1',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 5000, "pe": [], "pt": []  },
]

def mainRun(): 
    print("___Start!___" +  datetime.now().strftime('%H:%M:%S')  )
    final = "_" ;   md.DESC = "FRFLO";   ALL_DS = md.LOGDAT + md.DESC + md.DSC 
    
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
        mr.ninp, mr.nout = md.getnn()
        md.MODEL_DIR = md.LOGDIR + md.DESC + '/'   + mr.get_hpar(mr.epochs, final=final) +"/" 
        mr.model_path = md.MODEL_DIR + "model.ckpt" 
        mr.build_network3()
        print(mr.model_path)    

        # mr.evaluate( )

        ex["pt"] = mr.tests(url_test, p_col=False  )

    print("end!___" +  datetime.now().strftime('%H:%M:%S')  )

    print("0<60_1>60__0<23_1<60_2<93_3>93  ")
    for i in range(20):
        print("m:{0:15} - R-{4:5}   ||    P1-{1:2}         P2-{2:2}        P3-{3}" 
        .format(md.dsp.iloc[i,0], execc[0]["pt"][1][i][0], execc[1]["pt"][1][i][0],  execc[2]["pt"][1][i], md.dsp.iloc[i,1], ))    

if __name__ == '__main__':
    mainRun()


def bk():
    for i in range(20):
        print("RealVal: {}  - PP value: {}".format( md.dc( md.dataE['label'][i]), 
                                                    md.dc( predv.tolist()[i], np.max(predv[i]))  ))
    gt3, gtM = md.check_perf_CN(softv, md.dataE, False) #predv


    range_ts = len(predv) if len(predv)<20 else 20
    for i in range( range_ts ):
        # print("RealVal: {}  - PP value: {}".format( md.dc( dataTest['label'][i]), md.dc( predv.tolist()[i], np.max(predv[i]))  ))  
        print("{} RealVal: {} - {} - PP: {} PR: {}".format( i, md.dc( dataTest['label'][i]), sf[1][i][0],  sf[1][i], sf[0][i]   ))




