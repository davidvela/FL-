# load the mRun and execute it with different models ... 
# save the results and give a estimation comparing all the model resutls'

import mRun as rm
import utils_data as md

# epochs   = 5 #100
# lr       = 0.001 #0.0001
# h      = [100 , 100]
executions = [
    {   'dType':'C2',  'DESC':'FRFLO', 'model': "slr_1E-03_NN1814x100x100x102_ep100_", "spn": 5000, "predEv": [], "predts": []  },
    {   'dType':'C4',  'DESC':'FRFLO', 'model': "slr_1E-03_NN1814x100x100x102_ep100_", "spn": 5000, "predEv": [], "predts": []  },
    {   'dType':'C1',  'DESC':'FRFLO', 'model': "slr_1E-03_NN1814x100x100x102_ep100_", "spn": 5000, "predEv": [], "predts": []  },
]

def mainRun(): 
    print("___Start!___" +  datetime.now().strftime('%H:%M:%S')  )

    for ex in executions:
        md.DESC       = "FRFLO"
        md.spn        = 5000  
        md.dType      = "C1" #C1, C2, C4
        md.MODEL_DIR = md.LOGDIR + md.DESC + '/'   + get_hpar(epochs, final=final) +"/" 

        print("___Data Read!")
        mr.model_path = md.MODEL_DIR + "model.ckpt" 
        mr.ninp, mr.nout    = md.mainRead()

        build_network3()
        print(model_path)    

        mr.evaluate( )
        mr.url_test = md.LOGDAT + "FREXP1/" ;
        mr.tests(url_test, p_col=False  )

    print("___Start!___" +  datetime.now().strftime('%H:%M:%S')  )

if __name__ == '__main__':
    mainRun()


def bk
    for i in range(20):
        print("RealVal: {}  - PP value: {}".format( md.dc( md.dataE['label'][i]), 
                                                    md.dc( predv.tolist()[i], np.max(predv[i]))  ))
    gt3, gtM = md.check_perf_CN(softv, md.dataE, False) #predv


    range_ts = len(predv) if len(predv)<20 else 20
    for i in range( range_ts ):
        # print("RealVal: {}  - PP value: {}".format( md.dc( dataTest['label'][i]), md.dc( predv.tolist()[i], np.max(predv[i]))  ))  
        print("{} RealVal: {} - {} - PP: {} PR: {}".format( i, md.dc( dataTest['label'][i]), sf[1][i][0],  sf[1][i], sf[0][i]   ))




