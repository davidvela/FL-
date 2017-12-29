# load the mRun and execute it with different models ... 
# save the results and give a estimation comparing all the model resutls'
from datetime import datetime
import mRun as mr
import utils_data as md

executions = [
{ 'dt':'C2', 'de':'FRFLO', "e":100, "lr":0.001, "h":[100 , 100], "spn": 5000, "pe": [], "pt": []  },
# { 'dt':'C4', 'de':'FRFLO', "e":100, "lr":0.001, "h":[100 , 100], "spn": 5000, "pe": [], "pt": []  },
# { 'dt':'C1', 'de':'FRFLO', "e":100, "lr":0.001, "h":[100 , 100], "spn": 5000, "pe": [], "pt": []  },
]

def mainRun(): 
    print("___Start!___" +  datetime.now().strftime('%H:%M:%S')  )
    final = "_"
    for ex in executions:
        md.DESC       = ex["de"]
        md.spn        = ex["spn"]  
        md.dType      = ex["dt"]
        epochs        = ex["e"]

        mr.ninp, mr.nout    = md.mainRead()
        md.MODEL_DIR = md.LOGDIR + md.DESC + '/'   + mr.get_hpar(epochs, final=final) +"/" 
        mr.model_path = md.MODEL_DIR + "model.ckpt" 
        mr.build_network3()
        print(mr.model_path)    

        # mr.evaluate( )
        url_test = md.LOGDAT + "FREXP1/" ; # url_test = "url"
        mr.tests(url_test, p_col=True  )

    print("end!___" +  datetime.now().strftime('%H:%M:%S')  )

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




