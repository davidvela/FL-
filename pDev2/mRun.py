# tensorboard --logdir=.\_zfp\data\my_graph
# tensorboard => http://localhost:6006 
# jupyter => http://localhost:8889
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import sys
import os
import time
from types import *
from collections import Counter
from datetime import datetime
import utils_data as md
# import pickle
import itertools

def get_nns(): 
    #nns =  str(ninp)+'*'+str(h[0])+'*'+str(h[1])+'*'+str(nout)
    nns =  str(ninp)+'x' 
    for i in range(len(h)):
        nns = nns +str(h[i])+'x'
    return nns +str(nout)
#script hpar (s)
def get_hpar(ep=100, final="_"): return "slr_%.0E_NN%s_ep%s%s" % (lr, get_nns(),str(ep),final)
def logr(datep = '' , time='', it=1000, nn='', typ='TR', DS='', AC=0, num=0, AC3=0, AC10=0, desc='', startTime=''):
    if desc == '': print("Log not recorded"); return 
    LOG = "../../LOGT2.txt"
    f= open(LOG ,"a+") #w,a,
    if datep != '':   dats = datep
    else:             dats = datetime.now().strftime('%d.%m.%Y') 
    if time != '':    times = time
    else:             times = datetime.now().strftime('%H:%M:%S') 

    line =  datetime.now().strftime('%d.%m.%Y') + '\t' + times # time C1,C2
    #v1
    # line = line + '\t' + str(it) + '\t'+  get_nns() +  '\t' + str(lr) # IT NN LR TYP 
    #v2
    line = line + '\t' +  get_hpar(epochs, final=final) + '\t'+ '\t' 
    # values 
    line = line + '\t' + typ + '\t' + str(DS) # type + domain 
    line = line + '\t' + str(AC) + '\t' + str(num) + '\t' + str(AC3) + '\t' +  str(AC10) + '\t' + desc 
    line = line + '\t' + str(batch_size) + '\t' +  startTime + '\n' #new

    f.write(line);  f.close()
    print("___Log recorded")    
def restore_model(sess):   
    saver= tf.train.Saver() 
    print("Model restored from file: %s" % model_path)
    saver.restore(sess, model_path)

# NETWORK-----------------------------------------------------
def fc(inp, nodes, kp, is_train):
    # h = tf.layers.dense( x, h[0], activation=tf.nn.relu,  name )
    h = tf.layers.dense( inp, nodes, use_bias=False, activation=None )
    if md.dType != "C0": h = tf.layers.batch_normalization(h, training=is_train)      # CLASS
    h = tf.nn.relu(h)
    h = tf.nn.dropout(h, kp)
    return h
def build_network2(is_train=False):     # Simple NN - with batch normalization (high level)
    global top_k

    kp = 0.5
    inp = x
    # h0 = fc(x,  h[0], kp, is_train)
    # h1 = fc(h0, h[1], kp, is_train)    
    for i in range(len(h)): 
        hx = fc(inp,  h[i], kp, is_train); inp = hx 
    out = tf.layers.dense( hx, nout, use_bias=False, activation=None )
    prediction=tf.reduce_max(y,1)    # CLASS
    # prediction = out                 # REG

    # softmaxT = tf.nn.softmax(out)
    with tf.name_scope("accuracy"):
        softmaxT = tf.nn.top_k(tf.nn.softmax(out), top_k)                       # CLASS
        # values, indices  = tf.nn.top_k(tf.nn.softmax(out), top_k)             # CLASS
        # prediction_classes = table.lookup(tf.to_int64(indices))
        correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))       # CLASS
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))      # CLASS

        if md.dType == "C0":
            total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))        # REG
            unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, out)))                # REG
            accuracy = tf.subtract(tf.to_float(1), tf.div(total_error, unexplained_error))   # REG

        tf.summary.scalar("accuracy", accuracy)

    return out, accuracy, softmaxT
def build_network3():
    tf.reset_default_graph()

    global prediction, accuracy, softmaxT, cost, summ, optimizer, saver, x, y, confusion

    print("build network")
    x = tf.placeholder(tf.float32,   shape=[None, ninp], name="x")
    # y = tf.placeholder(tf.int16,     shape=[None, nout], name="y")
    y = tf.placeholder(tf.float32,     shape=[None, nout], name="y")
    prediction, accuracy, softmaxT = build_network2()
    
    # confusion = tf.confusion_matrix(labels=y, predictions=prediction, num_classes=nout)
    
    with tf.name_scope("xent"): #loss
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
        if md.dType == "C0": cost = tf.reduce_mean(tf.square(prediction-y) )               # REG
        tf.summary.scalar("xent", cost)
    
    with tf.name_scope("train"): #optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    
    summ = tf.summary.merge_all()
    saver= tf.train.Saver()

def build_network1( ):
    # Simple NN - 2layers - matmul 
    biases  = { 'b1': tf.Variable(tf.random_normal( [ h[0] ]),        name="Bias_1"),
                'b2': tf.Variable(tf.random_normal( [ h[1] ]),        name="Bias_2"),
                'out': tf.Variable(tf.random_normal( [nout] ),        name="Bias_out") }
    weights = { 'h1': tf.Variable(tf.random_normal([ninp,h[0]]),      name="Weights_1"),
                'h2': tf.Variable(tf.random_normal([h[0],h[1]]),      name="Weights_2"),
                'out': tf.Variable(tf.random_normal([h[1], nout]),    name="Weights_out")}

    # tf.reset_default_graph( )
    with tf.name_scope("fc_1"):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    with tf.name_scope("fc_2"):
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    with tf.name_scope("fc_output"):
        out = tf.matmul(layer_2, weights['out']) + biases['out']

    softmaxT = tf.nn.softmax(out, )
    prediction=tf.reduce_max(y,1)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    return out, accuracy, softmaxT, biases, weights
def old():
    # prediction, accuracy, softmaxT, biases, weights = build_network1()
    prediction, accuracy, softmaxT = build_network2()

    with tf.name_scope("xent"): #loss
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        tf.summary.scalar("xent", cost)
    with tf.name_scope("train"): #optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    summ = tf.summary.merge_all()
    saver= tf.train.Saver()

# OPERATIONS-----------------------------------------------------
def train(it = 100, disp=50, batch_size = 128, compt = False): 
    print("____TRAINING...")
    display_step =  disp 

    dataTest = {'label' : [] , 'data' :  [] };
    if compt: 
        md.get_columns(pp_excel = True )  # True! #md.dsc or dataset? 
        dataTest['data'] = md.dsc.iloc[:, 3:].as_matrix().tolist(); dataTest['label'] = md.dsc.iloc[:, 2].as_matrix().tolist()

        
    print("data read - lenTrain={}-{} & lenEv={}-{}, col = {}" 
            # .format(len(md.dataT["data"]), len(md.dataT["label"]),len(md.dataE["data"]),len(md.dataE["label"]) ))
            .format(len(md.dst.iloc[md.spn:, 3:]), len(md.dst.loc[md.spn:,'FP_P']),
            len( md.dst.iloc[:md.spn, 3:] ),len( md.dst.loc[:md.spn-1,'FP_P'] ), 
            len(dataTest['data'])
             ))
    
    total_batch  = int(len(md.dataT['label']) / batch_size)   
    startTime = datetime.now().strftime('%H:%M:%S')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # restore_model(sess)  #Run if I want to retrain an existing model  
        writer = tf.summary.FileWriter(md.MODEL_DIR + "/tboard/", sess.graph ) # + get_hpar() )

        start = time.time()
        for i in range(it):            
            for ii, (xtb,ytb) in enumerate(md.get_batches(batch_size) ):
                # xtb, ytb = dc.next_batch(batch_size, dataT['data'], dataT['label'])
                sess.run(optimizer, feed_dict={x: xtb, y: ytb})
                if ii % display_step ==0: #record_step == 0:
                    #[train_accuracy] = sess.run([accuracy], feed_dict={x: xtb, y: ytb })
                    # s = sess.run(summ, feed_dict={x: xtb, y: ytb })
                    [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: xtb, y: ytb }) 
                    writer.add_summary(s, i)
                     
                    elapsed_time = float(time.time() - start)
                    reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
                    rp_s = str(reviews_per_second)[0:5]
                    tr_ac = str(train_accuracy)[:5]  
                    print('Epoch: {} batch: {} / {} - %Speed(it/disp_step): {} - tr_ac {}' .format(i, ii, total_batch, rp_s, tr_ac ))
                    # writer.add_summary(s, i)
            test_accuracy = sess.run( accuracy, feed_dict={ x: md.dst.iloc[:md.spn, 3:],  y: md.dst.loc[:md.spn-1,'FP_P'].as_matrix().tolist()  })
            ev_ac = str(test_accuracy)[:5]  
            print("E Ac:", ev_ac)
            
            if compt: 
                sess.run([optimizer], feed_dict={x: dataTest['data'], y: dataTest['label']})
                tr_ac = str(sess.run(accuracy, feed_dict={x: dataTest['data'], y: dataTest['label']}))[:5] 
                print("Cm Ac:", tr_ac)
            
            train_accuracy = sess.run( accuracy, feed_dict={ x: md.dst.iloc[md.spn:, 3:],  y: md.dst.loc[md.spn:,'FP_P'].as_matrix().tolist()   })
            tr_ac = str( train_accuracy )[:5] 
            print("T Ac:", tr_ac)

            train_accuracies.append(tr_ac)
            test_accuracies.append(ev_ac)

        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path) 
    print("Optimization Finished!")

    logr( it=it, typ='TR', DS=md.DESC, AC=tr_ac,num=len(md.dst)-md.spn, AC3=0, AC10=0, desc=md.des(), startTime=startTime )
    logr( it=it, typ='EV', DS=md.DESC, AC=ev_ac,num=md.spn, AC3=0, AC10=0, desc=md.des() )
    dataTest = {'label' : [] , 'data' :  [] };


def train_opt( ): 
    pass

def evaluate( ): 
    print("_____EVALUATION...")
    startTime = datetime.now().strftime('%H:%M:%S')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restore_model(sess)
        # test the model
        tr_ac = str(sess.run( accuracy, feed_dict={ x: md.dst.iloc[md.spn:, 3:],  y: md.dst.loc[md.spn:,'FP_P'].as_matrix().tolist()    }) )[:5]  
        ev_ac = str(sess.run( accuracy, feed_dict={ x: md.dst.iloc[:md.spn, 3:],  y: md.dst.loc[:md.spn-1,'FP_P'].as_matrix().tolist()  }))[:5] 
        print("Training   Accuracy:", tr_ac )
        print("Evaluation Accuracy:", ev_ac )
        # xtp1.append(dataTest['data'][i]);    ytp1.append(dataTest['label'][i])
        predv, sf = sess.run([prediction, softmaxT], feed_dict={x: md.dst.iloc[:md.spn, 3:]  }) # , y: md.dataE['label'] 
        # maxa = sess.run([prediction], feed_dict={y: predv })
        
    print("Preview the first predictions:")
    for i in range(20):
        print("RealVal: {}  - PP value: {}".format( md.dc( md.dst.loc[:md.spn-1,'FP_P'][i])   , 
                                                    md.dc( predv.tolist()[i], np.max(predv[i]))  ))
    gt3, gtM = md.check_perf_CN(sf, md.dst.loc[:md.spn-1,'FP_P'], False)
    logr(  it=epochs, typ='EV', AC=ev_ac,DS=md.DESC, num=md.spn, AC3=gt3, AC10=gtM, desc=md.des(), startTime=startTime )
    
    calc_confusion_m( sf, md.dst.loc[:md.spn-1,'FP_P'], "EV")

    return predv.tolist()

def tests(url_test = 'url', p_col=False):  
    print("_____TESTS...")    
    
    # Load test data 
    dataTest = {'label' : [] , 'data' :  [] }; pred_val = []
    if p_col:                   # test columns 
        md.get_columns( )  #md.dsc
        dataTest['data'] = md.dsc.iloc[:, 3:].as_matrix().tolist(); dataTest['label'] = md.dsc.iloc[:, 2].as_matrix().tolist()
    elif p_col == False:        # test dataset 
        # logic migrated to get_tests -- 04.02
        # if url_test != 'url':   # test  file 
        #     json_data = url_test + "data_jsonX.txt"
        #     tmpLab = pd.read_csv(url_test + "datalX.csv", sep=',', usecols=[0,1])    
        #     tmpLab = tmpLab.loc[:,'fp']
        #     DESC   = "FREXP1_X"
        # else:                   # get data test JSON = url
        #     json_str, tmpLab = md.get_data_test(md.DESC)
        #     json_data = json.loads(json_str)
        #     DESC =  'matnrList...'
        # force = False

        md.get_tests(url_test ) #dsp
        dataTest['data']  = md.dsp.iloc[:, 3:].as_matrix().tolist(); dataTest['label'] = md.dsp.iloc[:, 2].as_matrix().tolist()     
    # Predict data 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restore_model(sess)
        # predv = sess.run( prediction, feed_dict={x: dataTest['data']}) 
        ts_acn = '0'
        ts_acn, predv, sf = sess.run( [accuracy, prediction, softmaxT], feed_dict={x: dataTest['data'], y: dataTest['label']}) 
        ts_ac = str(ts_acn) 
        print("test ac = {}".format(ts_ac))
    
    # print(dataTest['label']);     print(sf)
    range_ts = len(predv) if len(predv)<20 else 20
    for i in range( range_ts ):
        # print("RealVal: {}  - PP value: {}".format( md.dc( dataTest['label'][i]), md.dc( predv.tolist()[i], np.max(predv[i]))  ))  
        print("{} RealVal: {} - {} - PP: {} PR: {}".format( i, md.dc( dataTest['label'][i]), sf[1][i][0],  sf[1][i], sf[0][i]  ))

    # return
    gt3, gtM = md.check_perf_CN(sf, dataTest["label"], False)
    logr( it=0, typ='TS', DS=md.DESC, AC=ts_acn ,num=len(dataTest["label"]),  AC3=gt3, AC10=gtM, desc=md.des() )  

    calc_confusion_m( sf, md.dsp["FP_P"], "TS" )

    # outfile = md.LOGDAT + 'export2' 
    # np.savetxt(outfile + '.csv', sf[1], delimiter=',')
    # np.savetxt(outfile + 'PRO.csv', sf[0], delimiter=',')
    dataTest = {'label' : [] , 'data' :  [] }; pred_val = []
    return sf

def tests_exec(url_test = 'url', ret_str=True):  
    dataTest = {'label' : [] , 'data' :  [] }; pred_val = []
    # md.get_tests(url_test) #dsp
    tmpLab = [1] # dummy 
    json_str = url_test
    # print(json_str)
    if json_str[0]=="'": 
        json_str = json_str[1:]
        if json_str[len(json_str)-1]=="'": json_str = json_str[:len(json_str)-1]
        json_str = json_str.replace("'", '"')
    json_str = "[" + json_str + "]" 
    # json_str =   '''{ "m":"1", "100023" : 1 }  ''';  json_str = "[" + json_str + "]" 
    # print(json_str)
    json_data = json.loads(json_str)
    # print(json_data)
    md.dsp = md.feed_data(json_data ,pand=True, d_st=True, p_all = True)       #d_st = display status
    ds = md.dsp.iloc[0]
    if len(ds.iloc[ds.nonzero()]) == 1: return "error: comp 0"
    md.dsp["FP"] = tmpLab; 
    md.dsp.insert(2, 'FP_P', md.dsp['FP'].map(lambda x: md.cc( x )))
    dataTest['data']  = md.dsp.iloc[:, 3:].as_matrix().tolist();   # No FP
    dataTest['label'] = md.dsp.iloc[:, 2].as_matrix().tolist()     # dummy 
     
    # md.print_form2(dsp.iloc[0]);     # print(dsp.iloc[0])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restore_model(sess)
        # predv = sess.run( prediction, feed_dict={x: dataTest['data']}) 
        predv, sf = sess.run( [ prediction, softmaxT], feed_dict={x: dataTest['data']}) #, y: dataTest['label']}) 

    for i in range( 1 ):
        # print("RealVal: {}  - PP value: {}".format( md.dc( dataTest['label'][i]), md.dc( predv.tolist()[i], np.max(predv[i]))  ))  
        print("{} RealVal: {} - {} - PP: {} PR: {}".format( i, md.dc(dataTest['label'][i]), sf[1][i][0],  sf[1][i], sf[0][i]  ))
    
    logr( it=0, typ='AP', DS=md.DESC, AC=sf[1][0] ,num=sf[0][0],  AC3=0, AC10="real?", desc=str(md.dsp.iloc[0,0] ) )  
    # logr( it=0, typ='TS', DS=md.DESC, AC=ts_acn ,num=len(dataTest["label"]),  AC3=gt3, AC10=gtM, desc=md.des() )  

    # if ret_str: return "PP: {} PR: {}".format(sf[1][0], sf[0][0] )
    if ret_str: return "PP: {} PR: {}".format(sf[1][0], sf[0][0] )
    else: return md.get_json_format(md.dType, sf[1][0], sf[0][0] )

def clean_traina():
    global train_accuracies, test_accuracies
    train_accuracies, test_accuracies = [], []
    
def vis_chart( ):
    fig, ax = plt.subplots()
    plt.plot(train_accuracies, label='Train', alpha=0.5)
    plt.plot(test_accuracies, label='Test', alpha=0.5)
    plt.title("Accuracy" + md.MODEL_DIR)
    plt.legend()
    # plt.savefig(md.MODEL_DIR + "/chart.png" )
    plt.savefig(md.MODEL_DIR + ".png" )
    # plt.show()
    return

def vis_confusion_m(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, 
                          tid="t"):
    plt.figure()
    if md.dType == "C1": 
        fig = plt.gcf()
        fig.set_size_inches(32, 32)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(md.LOGDAT + md.dType + "_" + tid + "conf_mat.png" )
    # plt.clf()
    # plt.cla()  # for multiple subplots


iccm = 0 
def calc_confusion_m( sf, dst, tid="t"):
    global iccm 
    confusion = tf.confusion_matrix(    labels=md.get_conv_list(  dst ), 
                                        predictions=[ sf[1][x][0]  for x in range( len(sf[1]) )   ], 
                                        num_classes=nout)
    with tf.Session() as sess:
        conf = sess.run(confusion)
   
    np.savetxt(md.LOGDAT + "cm" + tid +".csv", conf, delimiter=",")
    
    if md.dType == "C1N": 
        print(conf)
    else: 
        class_names = [str(i) for i in range(md.nout)]
        vis_confusion_m( conf, classes=class_names, normalize=False,
                            title='Confusion matrix, without normalization',
                            cmap=plt.cm.cool, tid=tid)

    


#--------------------------------------------------------------
md.DESC      = "FLALL" # FLALL "FREXP"  FRFLO FRALL1 || #C1, C2, C4, C0 || #[40 , 10]   [200, 100, 40] [100,100]
ex =  { 'dt':'C4',  "e":100, "lr":0.001, "h":[40 , 40],       "spn": 10000, "pe": [], "pt": []  }
# ex =  { 'dt':'C1',  "e":200, "lr":0.001, "h":[100 , 100, 100], "spn": 10000, "pe": [], "pt": []  }

md.spn       = ex["spn"] 
md.dType     = ex["dt"] 
epochs       = ex["e"] 
lr           = ex["lr"]
h            = ex["h"]   

ninp, nout   = 10, 10
disp         = 5
batch_size   = 128
final        = "_" #FF or _

def mainRun(): 
    global ninp, nout, model_path, top_k
    print("___Start!___" +  datetime.now().strftime('%H:%M:%S')  )
    #---------------------------------------------------------------
    # DATA READ 
    #---------------------------------------------------------------
    ALL_DS     = md.LOGDAT + md.DESC + md.DSC 
    md.setDESC(md.DESC)
    md.mainRead2(ALL_DS, 1, 2, all = True, shuffle = True  ) 
    # md.mainRead2(ALL_DS, 1, 2, all = False ) # For testing I am forced to used JSON - column names and order may be different! 
    md.normalize()
    ninp, nout, top_k = md.getnn()
    # print(len(md.dst))
    md.MODEL =  get_hpar(epochs, final=final)
    md.MODEL_DIR = md.LOGDIR + md.DESC + '/' + md.MODEL  #+"/" 
    model_path = md.MODEL_DIR + "/model.ckpt" 
    force = False        
    url_test = md.LOGDAT + "FREXP1/" ; # url_test = "url"
    # md.get_tests(url_test=url_test, force=force, pp_excel=True)
    md.get_columns(force, True) # True => read from excel...

    #---------------------------------------------------------------
    # NETWORK
    #---------------------------------------------------------------
    build_network3()
    print(model_path)
    print( get_nns() )
    clean_traina()
    
    #---------------------------------------------------------------
    # OP.                           comp. 
    #---------------------------------------------------------------

    train(epochs, disp, batch_size, True)
    evaluate( )
    
    # tests(url_test, p_col=False  )
    vis_chart( )  # visualize the training chart
    print("___The end!")

    #---------------------------------------------------------------
    # NN                           retrain! - optimization... 
    #---------------------------------------------------------------
    # optimize - old_model, new_model, --- why? how?? 
    #   take the wrong data and retrain the model with only the errors ... 
    #   modify the train function - optimization 


def mainOPT(): 
    pass

if __name__ == '__main__':
    mainRun()




