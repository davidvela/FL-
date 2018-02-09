#tensorflow serving - export model - protobuf 

from datetime import datetime
import tensorflow as tf
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
            { 'dt':'C2',  "e":40,  "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
            { 'dt':'C4',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
            { 'dt':'C1',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
            # { 'dt':'C0',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
        ]
    else: return []

def mainRun(): 
    print("___Start!___" +  datetime.now().strftime('%H:%M:%S')  )
    final = "_" ;  md.DESC = "FRFLO";  # FRFLO   FRALL1
    ALL_DS = md.LOGDAT + md.DESC + md.DSC 
    execc = get_models(md.DESC)

    # -------------------------------------------------------------
    # DATA READ  
    # -------------------------------------------------------------
    # md.mainRead2(ALL_DS, 1, 2 ) # , all = True, shuffle = True  ) 
    # url_test = md.LOGDAT + "FREXP1/" ; # url_test = "url"
    # force = False; excel = True  # dataFile = "frall2_json.txt"; labelFile = "datal.csv"     
    # md.get_tests(url_test, force, excel )

    # -------------------------------------------------------------
    # READ MODEL  
    # -------------------------------------------------------------
    # md.get_columns(force)   # for ex in execc:
    ex = execc[2]
    # md.spn = ex["spn"]; md.dType = ex["dt"]; mr.epochs = ex["e"]; mr.lr = ex["lr"]; mr.h = ex["h"] 
    # md.normalize()
      
    mr.ninp = 1814
    mr.ninp, mr.nout, mr.top_k = md.getnn(mr.ninp)
    
    md.MODEL_DIR = md.LOGDIR + md.DESC + '/'   + mr.get_hpar(mr.epochs, final=final) +"/" 
    mr.model_path = md.MODEL_DIR + "model.ckpt" 
    mr.build_network3()                                                                                                                                                                                                                                                                                    
    print(mr.model_path)    
    # ex["pe"]   = mr.evaluate( )
    # ex["pt"]   = mr.tests(url_test, p_col=False  )
    
    # -------------------------------------------------------------
    # EXPORT MODEL  
    # -------------------------------------------------------------

    # sess = tf.InteractiveSession()
    with tf.Session() as sess:
        # Restore the model from last checkpoints
        # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        # saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(tf.global_variables_initializer())
        mr.restore_model(sess)
        
        export_path = "./export/" + mr.get_hpar(mr.epochs, final=final)
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        # Build the signature_def_map.

        classification_inputs = tf.saved_model.utils.build_tensor_info(mr.x)
        # classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
        # classification_outputs_scores = tf.saved_model.utils.build_tensor_info(mr.softmaxT)
        predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(mr.prediction)
        
        prediction_signature = ( 
            tf.saved_model.signature_def_utils.build_signature_def( 
                inputs={ tf.saved_model.signature_constants.CLASSIFY_INPUTS: classification_inputs },        
                outputs={tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: predict_tensor_scores_info},
                # outputs={'scores': predict_tensor_scores_info},
                # outputs={
                #       tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: classification_outputs_classes,
                #       tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:  classification_outputs_scores
                method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME)) 

        # - save... 
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op') 
        builder.add_meta_graph_and_variables(   
                    sess, 
                    [tf.saved_model.tag_constants.SERVING], 
                    signature_def_map={ 'predict_val': prediction_signature }, 
                    legacy_init_op=legacy_init_op) 
        builder.save()
    print('Done exporting!')

if __name__ == '__main__':
    mainRun()
    