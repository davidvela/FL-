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
import pickle as pkl

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
def clean_traina():
    global train_accuracies, test_accuracies, samples
    train_accuracies, test_accuracies, samples = [], [], []
    
# NETWORK GAN -------------- # -> not used variables    
def generator(z, output_dim, reuse=False, alpha=0.2, training=True, size_mult=128):
    is_train=False
    with tf.variable_scope('generator', reuse=reuse):
        # First fully connected layer
        x1 = tf.layers.dense(z, h[0], use_bias=False, activation=None )
        x1 = tf.layers.batch_normalization(x1, training=is_train)
        x1 = tf.nn.relu(x1)
        #x1 = tf.nn.dropout(x1, kp)
        x2 = tf.layers.dense( x1, h[1], use_bias=False, activation=None )
        x2 = tf.layers.batch_normalization(x2, training=is_train)
        h1 = tf.nn.relu(x2)
        # h1 = tf.nn.dropout(x2, kp)
        out = tf.layers.dense( h1, output_dim, use_bias=False, activation=None )
        return out   
    
extra_class = 0                   # -> not used variables    YES!
def discriminator(x, reuse=False, alpha=0.2, drop_rate=0., num_classes=10, size_mult=64):
    is_train=False
    with tf.variable_scope('discriminator', reuse=reuse):
        # First fully connected layer
        x1 = tf.layers.dense(x, h[0], use_bias=False, activation=None )
        x1 = tf.layers.batch_normalization(x1, training=is_train)
        x1 = tf.nn.relu(x1)
        #x1 = tf.nn.dropout(x1, kp)
        x2 = tf.layers.dense( x1, h[1], use_bias=False, activation=None )
        x2 = tf.layers.batch_normalization(x2, training=is_train)
        h1 = tf.nn.relu(x2)
        # h1 = tf.nn.dropout(x2, kp)
        features = x
        class_logits  = tf.layers.dense( h1, num_classes+extra_class, use_bias=False, activation=None )
        # out = tf.layers.batch_normalization(out, training=is_train)
        # out = tf.nn.relu(out)
        # out = tf.nn.dropout(out, kp)

        if extra_class:
            real_class_logits, fake_class_logits = tf.split(class_logits, [num_classes, 1], 1)
            assert fake_class_logits.get_shape()[1] == 1, fake_class_logits.get_shape()
            fake_class_logits = tf.squeeze(fake_class_logits)
        else:
            real_class_logits = class_logits
            fake_class_logits = 0.
            
        mx = tf.reduce_max(real_class_logits, 1, keep_dims=True)
        stable_real_class_logits = real_class_logits - mx
        gan_logits = tf.log(tf.reduce_sum(tf.exp(stable_real_class_logits), 1)) + tf.squeeze(mx) - fake_class_logits
        out = tf.nn.softmax(class_logits)
        return out, class_logits, gan_logits, features
        
def model_loss(input_real, input_z, output_dim, y, num_classes, label_mask, alpha=0.2, drop_rate=0.):
    g_size_mult = 32  # NOT USED
    d_size_mult = 64  # NOT USED
    # run gen and disc
    g_model = generator(input_z, output_dim, alpha=alpha, size_mult=g_size_mult)
    d_on_data = discriminator(input_real, num_classes=num_classes, alpha=alpha, drop_rate=drop_rate, size_mult=d_size_mult)
    d_model_real, class_logits_on_data, gan_logits_on_data, data_features = d_on_data
    
    d_on_samples = discriminator(g_model,num_classes=num_classes, reuse=True, alpha=alpha, drop_rate=drop_rate, size_mult=d_size_mult)
    d_model_fake, class_logits_on_samples, gan_logits_on_samples, sample_features = d_on_samples
    
    # compute the losses. 
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_data,
                                                labels=tf.ones_like(gan_logits_on_data)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_samples,
                                                labels=tf.zeros_like(gan_logits_on_samples)))
    # # simple GAN - generator:
    # d_loss = d_loss_real + d_loss_fake
    # g_loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits( logits=d_logits_fake, labels=tf.ones_like(d_model_fake)  ))
    # return d_loss, g_loss

    # semi-supervised                                            
    y = tf.squeeze(y)
    class_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=class_logits_on_data,
                                                                  labels=tf.one_hot(y, num_classes + extra_class,
                                                                  dtype=tf.int32))
    class_cross_entropy = tf.squeeze(class_cross_entropy)
    label_mask = tf.squeeze(tf.to_float(label_mask))
    d_loss_class = tf.reduce_sum(label_mask * class_cross_entropy) / tf.maximum(1., tf.reduce_sum(label_mask))
    d_loss = d_loss_class + d_loss_real + d_loss_fake
    
    # set `g_loss` to the "feature matching" loss invented by Tim Salimans at OpenAI
    data_moments = tf.reduce_mean(data_features, axis=0)
    sample_moments = tf.reduce_mean(sample_features, axis=0)
    g_loss = tf.reduce_mean(tf.abs(data_moments - sample_moments))

    
    pred_class = tf.cast(tf.argmax(class_logits_on_data, 1), tf.int32)
    # eq = tf.equal(tf.squeeze(y), pred_class)                                # GAN
    eq = tf.equal(tf.argmax(class_logits_on_data, 1), tf.argmax(y, 1))      # CLASS
    correct = tf.reduce_sum(tf.to_float(eq))                                # GAN 
    masked_correct = tf.reduce_sum(label_mask * tf.to_float(eq))

    # classification 
    # softmaxT = tf.nn.top_k(tf.nn.softmax(out), top_k)                       # CLASS
    # correct = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))                  # CLASS
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))      # CLASS

    return d_loss, g_loss, correct, masked_correct, g_model

def model_opt(d_loss, g_loss, learning_rate, beta1):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    for t in t_vars:
        assert t in d_vars or t in g_vars
    d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
    shrink_lr = tf.assign(learning_rate, learning_rate * 0.9)
    return d_train_opt, g_train_opt, shrink_lr
z_size = 100
class GAN:
    def __init__(self, ninp, nout, top_k,  learning_rate, num_classes=10, alpha=0.2, beta1=0.5):
        tf.reset_default_graph()
        
        self.learning_rate = tf.Variable(learning_rate, trainable=False)
        # self.input_real, self.input_z, self.y, self.label_mask = model_inputs(real_size, z_size)        
        self.input_real    = tf.placeholder(tf.float32,   shape=[None, ninp], name="input_real")
        self.input_z       = tf.placeholder(tf.float32,   shape=[None, z_size], name="input_z")
        self.y              = tf.placeholder(tf.int32,    shape=[None, nout], name="y")
        self.label_mask     = tf.placeholder(tf.int32,    (None),             name='label_mask')
        self.drop_rate      = tf.placeholder_with_default(.5, (), "drop_rate")
        

        loss_results = model_loss(self.input_real, self.input_z, ninp, self.y, nout, 
            label_mask=self.label_mask, alpha=0.2, drop_rate=self.drop_rate) 
        
        self.d_loss, self.g_loss, self.correct, self.masked_correct, self.samples = loss_results       
        self.d_opt, self.g_opt, self.shrink_lr = model_opt(self.d_loss, self.g_loss, self.learning_rate, beta1)
        

print("net declared")

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
    prediction = out                 # REG

    # softmaxT = tf.nn.softmax(out)
    with tf.name_scope("accuracy"):
        softmaxT = tf.nn.top_k(tf.nn.softmax(out), top_k)                       # CLASS
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

    print("build network")
    global prediction, accuracy, softmaxT, cost, summ, optimizer, saver, x, y 
    x = tf.placeholder(tf.float32,   shape=[None, ninp], name="x")
    # y = tf.placeholder(tf.int16,     shape=[None, nout], name="y")
    y = tf.placeholder(tf.float32,     shape=[None, nout], name="y")
    prediction, accuracy, softmaxT = build_network2()
    
    with tf.name_scope("xent"): #loss
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        if md.dType == "C0": cost = tf.reduce_mean(tf.square(prediction-y) )               # REG
        tf.summary.scalar("xent", cost)
    
    with tf.name_scope("train"): #optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    summ = tf.summary.merge_all()
    saver= tf.train.Saver()
def restore_model(sess):   
    saver= tf.train.Saver() 
    print("Model restored from file: %s" % model_path)
    saver.restore(sess, model_path)
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
def get_input_z(): 
    sample_z = np.random.normal(0, 1, size=(z_size))
    # sample_z = np.random.normal(0, 1, size=(ninp))
    # mask_z = np.random.randint(2, size=ninp) ; sample_z= sample_z*mask_z
    return sample_z
def trainG(net, it = 100, disp=50, batch_size = 128, compt = False): 
    # def trainG(net, dataset, epochs, batch_size, figsize=(5,5)):
    print("____TRAINING...")
    display_step =  disp 

    # dataTest = {'label' : [] , 'data' :  [] };
    
    print("data read - lenTrain={}-{} & lenEv={}-{}" .format(len(md.dataT["data"]), len(md.dataT["label"]),len(md.dataE["data"]),len(md.dataE["label"]) ))
    total_batch  = int(len(md.dataT['label']) / batch_size)   
    startTime = datetime.now().strftime('%H:%M:%S')

    saver = tf.train.Saver()
    sample_z = get_input_z()
   
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # restore_model(sess)  #Run if I want to retrain an existing model  
        writer = tf.summary.FileWriter(md.MODEL_DIR + "tboard/", sess.graph ) # + get_hpar() )

        start = time.time()
        for i in range(it):            
            num_examples = 0; num_correct = 0;
            for ii, (xtb,ytb) in enumerate(md.get_batches(batch_size) ):
                # xtb, ytb = dc.next_batch(batch_size, dataT['data'], dataT['label'])
                batch_z = get_input_z()

                _, _, correct = sess.run([net.d_opt, net.g_opt, net.masked_correct],
                                         feed_dict={net.input_real: xtb, 
                                                    net.input_z: batch_z,
                                                    net.y : ytb
                                                    #,  net.label_mask : label_mask # mask to pretend to have unlabelled data
                                        }) 
                print(correct)
                # num_correct += correct
                num_examples += batch_size
                if ii % display_step ==0: #record_step == 0:
                    train_accuracy = num_correct / float(num_examples)
                    print("\t\tClassifier train accuracy: ", train_accuracy)
            
            num_examples = 0; num_correct = 0;
            # assert 'int' in str(y.dtype)
            num_examples = md.spn
            correct = sess.run( [net.correct], 
                feed_dict={ net.input_real: md.dst.iloc[:md.spn, 3:],  
                            net.y: md.dst.loc[:md.spn-1,'FP_P'].as_matrix().tolist(),
                            net.drop_rate: 0. })
            num_correct += correct
            ev_ac = num_correct / float(num_examples)
            print("E Ac:", ev_ac)
            
            correct = sess.run( [net.correct], 
                feed_dict={ x: md.dst.iloc[md.spn:, 3:],  
                            y: md.dst.loc[md.spn:,'FP_P'].as_matrix().tolist()   })
            tr_ac = num_correct / float(len(md.dst.iloc[md.spn:, 3:]))
            print("T Ac:", tr_ac)

            train_accuracies.append(train_accuracy)
            test_accuracies.append(ev_ac)

            gen_samples = sess.run( net.samples, feed_dict={net.input_z: sample_z})
            samples.append(gen_samples)


        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path) 
    print("Optimization Finished!")

    # logr( it=it, typ='TR', DS=md.DESC, AC=tr_ac,num=len(md.dst)-md.spn, AC3=0, AC10=0, desc=md.des(), startTime=startTime )
    # logr( it=it, typ='EV', DS=md.DESC, AC=ev_ac,num=md.spn, AC3=0, AC10=0, desc=md.des() )
    dataTest = {'label' : [] , 'data' :  [] };
    
    with open('samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
    
    return train_accuracies, test_accuracies, samples

def train(it = 100, disp=50, batch_size = 128, compt = False): 
    print("____TRAINING...")
    display_step =  disp 

    dataTest = {'label' : [] , 'data' :  [] };
    if compt: 
        md.get_columns(pp_excel = True )  #md.dsc or dataset? 
        dataTest['data'] = md.dsc.iloc[:, 3:].as_matrix().tolist(); dataTest['label'] = md.dsc.iloc[:, 2].as_matrix().tolist()
       
    print("data read - lenTrain={}-{} & lenEv={}-{}" .format(len(md.dataT["data"]), len(md.dataT["label"]),len(md.dataE["data"]),len(md.dataE["label"]) ))
    total_batch  = int(len(md.dataT['label']) / batch_size)   
    startTime = datetime.now().strftime('%H:%M:%S')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # restore_model(sess)  #Run if I want to retrain an existing model  
        writer = tf.summary.FileWriter(md.MODEL_DIR + "tboard/", sess.graph ) # + get_hpar() )

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
        predv, softv = sess.run([prediction, softmaxT], feed_dict={x: md.dst.iloc[:md.spn, 3:]  }) # , y: md.dataE['label'] 
        # maxa = sess.run([prediction], feed_dict={y: predv })
        
    print("Preview the first predictions:")
    for i in range(20):
        print("RealVal: {}  - PP value: {}".format( md.dc( md.dst.loc[:md.spn-1,'FP_P'][i])   , 
                                                    md.dc( predv.tolist()[i], np.max(predv[i]))  ))
    gt3, gtM = md.check_perf_CN(softv, md.dst.loc[:md.spn-1,'FP_P'], False)
    logr(  it=epochs, typ='EV', AC=ev_ac,DS=md.DESC, num=md.spn, AC3=gt3, AC10=gtM, desc=md.des(), startTime=startTime )
    return predv.tolist()
def tests(url_test = 'url', p_col=False):  
    print("_____TESTS...")    
    
    # Load test data 
    dataTest = {'label' : [] , 'data' :  [] }; pred_val = []
    if p_col:                   # test columns 
        md.get_columns( )  #md.dsc
        dataTest['data'] = md.dsc.iloc[:, 3:].as_matrix().tolist(); dataTest['label'] = md.dsc.iloc[:, 2].as_matrix().tolist()
    else: 
        if url_test != 'url':   # test  file 
            json_data = url_test + "data_jsonX.txt"
            tmpLab = pd.read_csv(url_test + "datalX.csv", sep=',', usecols=[0,1])    
            tmpLab = tmpLab.loc[:,'fp']
            DESC   = "FREXP1_X"
        else:                   # get data test JSON = url
            json_str, tmpLab = md.get_data_test(md.DESC)
            json_data = json.loads(json_str)
            DESC =  'matnrList...'
        force = False
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
    logr( it=0, typ='TS', DS=DESC, AC=ts_acn ,num=len(dataTest["label"]),  AC3=gt3, AC10=gtM, desc=md.des() )  

    # outfile = md.LOGDAT + 'export2' 
    # np.savetxt(outfile + '.csv', sf[1], delimiter=',')
    # np.savetxt(outfile + 'PRO.csv', sf[0], delimiter=',')
    dataTest = {'label' : [] , 'data' :  [] }; pred_val = []
    return sf

def vis_chart( ):
    fig, ax = plt.subplots()
    plt.plot(train_accuracies, label='Train', alpha=0.5)
    plt.plot(test_accuracies, label='Test', alpha=0.5)
    plt.title("Accuracy" + md.MODEL_DIR)
    plt.legend()
    plt.savefig(md.MODEL_DIR + "chart.png" )
    # plt.show()
    return

md.DESC      = "FRFLO" # "FREXP"  FRFLO
md.spn       = 10000  
md.dType     = "C4" #C1, C2, C4, C0
epochs       = 100 #100

lr           = 0.001 #0.0001
h            = [100 , 100]   #[40 , 10]   [200, 100, 40] [100,100]
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
    md.mainRead2(ALL_DS, 1, 2, all = True, shuffle = True  ) 
    # md.mainRead2(ALL_DS, 1, 2, all = False ) # For testing I am forced to used JSON - column names and order may be different! 
    md.normalize()
    ninp, nout, top_k = md.getnn()
    # print(len(md.dst))
    md.MODEL_DIR = md.LOGDIR + md.DESC + '/'   + get_hpar(epochs, final=final) +"/" 
    model_path = md.MODEL_DIR + "model.ckpt" 
    force = False        
    url_test = md.LOGDAT + "FREXP1/" ; # url_test = "url"
    md.get_tests(url_test=url_test, force=force, pp_excel=True)
    md.get_columns(force, True)

    #---------------------------------------------------------------
    # NETWORK
    #---------------------------------------------------------------
    # build_network3()
    net = GAN(ninp, nout, top_k, lr)


    print(model_path)
    print( get_nns() )
    clean_traina()
    
    #---------------------------------------------------------------
    # OP.                           comp. 
    #---------------------------------------------------------------
    train_accuracies, test_accuracies, samples = trainG(net, epochs, disp, batch_size, True  )
    # train(epochs, disp, batch_size, True)
    # evaluate( )
    # tests(url_test, p_col=False  )
    # vis_chart( )

    print("___The end!")

if __name__ == '__main__':
    mainRun()




