from flask import Flask, request
from flask_restplus import Api, Resource, fields
from werkzeug.contrib.fixers import ProxyFix
from datetime import datetime
import mRun as mr
import utils_data as md

app = Flask(__name__)
def app_config():
    pass
    # not working!!!! 
    # app.config['SERVER_NAME'] = '127.0.0.1' + ':' + '5002'
    # flask_app.config['SERVER_NAME'] = server_name + ':' + server_port
    # flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = settings.RESTPLUS_SWAGGER_UI_DOC_EXPANSION # 'list'
    # flask_app.config['RESTPLUS_VALIDATE'] = settings.RESTPLUS_VALIDATE # True
    # flask_app.config['RESTPLUS_MASK_SWAGGER'] = settings.RESTPLUS_MASK_SWAGGER # False
    # flask_app.config['ERROR_404_HELP'] = settings.RESTPLUS_ERROR_404_HELP # False 
    #                   - If a request does not match any of the application endpoints => return error 404 or not 

app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0', title='FP API',   description='Prototype to Predict FP API', )
ns = api.namespace('fp', description='FP Operations')

#model _______________________________
fp = api.model('fp', {
    #'id': fields.Integer(readOnly=True, description='The fp unique identifier'),
    'FORML'    : fields.String(required=True, description='The form details'),
    'RET_STR'  : fields.String(required=True, description='return string or not'),
    'PRED'     : fields.String(required= False,description='Prediction C2')  ,   #all - basic
    'PRED_C2_0': fields.String(required= False,description='Prediction C2')  ,
    'PRED_C2_1': fields.String(required= False,description='Prediction C2')  ,
    'PRED_C4_0': fields.String(required= False,description='Prediction C4')  ,
    'PRED_C4_1': fields.String(required= False,description='Prediction C4')  ,
    'PRED_C4_2': fields.String(required= False,description='Prediction C4')  ,
    'PRED_C1_0': fields.String(required= False,description='Prediction C1')  ,
    'PRED_C1_1': fields.String(required= False,description='Prediction C1')  ,
    'PRED_C1_2': fields.String(required= False,description='Prediction C1')  ,
    'PRED_C1_3': fields.String(required= False,description='Prediction C1')  ,
    'PRED_C1_4': fields.String(required= False,description='Prediction C1')  })

def get_models(type):
    if type == "FRFLO":
        return [
            { 'dt':'C2',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 5000, "pe": [], "pt": [], "ninp":1814  },
            { 'dt':'C4',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 5000, "pe": [], "pt": [], "ninp":1814  },
            { 'dt':'C1',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 5000, "pe": [], "pt": [], "ninp":1814  },
        ]
    elif type == "FRALL1":
        return [
            { 'dt':'C2',  "e":40,  "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": [], "ninp":2385  },
            { 'dt':'C4',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": [], "ninp":2385  },
            { 'dt':'C1',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": [], "ninp":2385  },
            # { 'dt':'C0',  "e":100, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": []  },
        ]
    elif type == "FRALL11":
            return [
            { 'dt':'C2',  "e":20, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": [], "ninp":2385  },
            { 'dt':'C4',  "e":50, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": [], "ninp":2385  },
            { 'dt':'C1',  "e":50, "lr":0.001, "h":[100 , 100], "spn": 10000, "pe": [], "pt": [], "ninp":2385  },
        ]
    else: return []


class fpDAO(object):
    def __init__(self):
        self.counter = 0
        self.todos = []

    def get(self, forml, ret_str = True ):
        print("___Start!___" +  datetime.now().strftime('%H:%M:%S')  )

        md.DESC = "FRALL1"; # FRFLO / FRALL1
        md.setDESC(md.DESC) 
        
        mr.final = "_" #_ _101
        execc = get_models(md.DESC);   
        
        # ex = execc[2]
        if ret_str:  rett = []
        else: rett = {}
        
        for ex in execc: 
            md.spn = ex["spn"]; md.dType = ex["dt"]; mr.epochs = ex["e"]; mr.lr = ex["lr"]; mr.h = ex["h"]
            mr.ninp = ex["ninp"]
            mr.ninp, mr.nout, mr.top_k = md.getnn(mr.ninp)
            
            md.MODEL_DIR = md.LOGDIR + md.DESC + '/'   + mr.get_hpar(mr.epochs, final=mr.final) 
            mr.model_path = md.MODEL_DIR + "/model.ckpt" 
            mr.build_network3()                                                                                                                                                                                                                                                                                    
        
            print(mr.model_path)    
            # forml2 = '{ "m":"1", "100023" : 1 }'
            # forml2 = '{ "m":"PBV10476AS", "178583" :0.74598 , "106104" :0.1 , "182789" :0.04 , "130172" :0.035 , "179661" :0.035 , "164421" :0.018 , "600040" :0.0108 , "116165" :0.008 , "164419" :0.0018 , "103396" :0.001 , "130217" :0.001 , "131460" :0.001 , "690750" :0.0007 , "611089" :0.0006 , "130058" :0.0004 , "130354" :0.0002 , "131101" :0.0002 , "131435" :0.00012 , "131136" :0.0001 , "131315" :0.0001 }'
            # forml2 = '{ "m":"607654", "130966" :0.32311827956989 , "903441" :0.16129032258065 , "131030" :0.10752688172043 , "160102" :0.07480258064516 , "660408" :0.05537634408602 , "168292" :0.05376344086022 , "172338" :0.03225806451613 , "106114" :0.02150795698925 , "130217" :0.02150537634409 , "198070" :0.01612903225806 , "658292" :0.01612903225806 , "173279" :0.01075268817204 , "103165" :0.01075268817204 , "184367" :0.00861075268817 , "613031" :0.00806451612903 , "131323" :0.00645161290323 , "130382" :0.00645161290323 , "673561" :0.00618279569892 , "130520" :0.00537634408602 , "106215" :0.00537634408602 , "103026" :0.00537634408602 , "103908" :0.00537634408602 , "130742" :0.00430107526882 , "130935" :0.00322580645161 , "627665" :0.00322580645161 , "195933" :0.00322580645161 , "600622" :0.00322580645161 , "130813" :0.00268817204301 , "164419" :0.00215309247312 , "103936" :0.00215053763441 , "102021" :0.00161290322581 , "130879" :0.00161290322581 , "103208" :0.00107526881720 , "173525" :0.00107526881720 , "160189" :0.00075436453768 , "613018" :0.00075268817204 , "619501" :0.00075268817204 , "131381" :0.00064516129032 , "115262" :0.00060169552410 , "130837" :0.00059139784946 , "105481" :0.00053763440860 , "660567" :0.00043010752688 , "103959" :0.00043010752688 , "710757" :0.00026881720430 , "100085" :0.00021505376344 , "103258" :0.00021505376344 , "164430" :0.00021505376344 , "130085" :0.00021505376344 , "100910" :0.00021505376344 , "657691" :0.00018859113442 , "659609" :0.00010752688172 , "130301" :0.00010752688172 , "690012" :0.00010752688172 , "105543" :0.00008602150538 , "104087" :0.00008172043011 , "131176" :0.00007526881720 , "613125" :0.00006825202960 , "601344" :0.00005376344086 , "131104" :0.00005376344086 , "130141" :0.00005376344086 , "130870" :0.00004301075269 , "107796" :0.00004301075269 , "113554" :0.00004179569892 , "195759" :0.00003225806452 , "656019" :0.00003225806452 , "131185" :0.00003225806452 , "164076" :0.00003225806452 , "103396" :0.00002150537634 , "103383" :0.00001935483871 , "104354" :0.00001612903226 , "105703" :0.00001354838710 , "638261" :0.00001075268817 , "103798" :0.00001075268817 , "105188" :0.00001075268817 , "608025" :0.00001075268817 , "657159" :0.00001010752688 , "600001" :0.00000690322581 , "660783" :0.00000645161290 , "600194" :0.00000537634409 , "131157" :0.00000430107527 , "195564" :0.00000322580645 , "690986" :0.00000322580645 , "100204" :0.00000258064516 , "660240" :0.00000215053763 , "100106" :0.00000193548387 , "127988" :0.00000129032258 , "198022" :0.00000107526882 , "173125" :0.00000086021505 , "130188" :0.00000086021505 , "173452" :0.00000075268817 , "103190" :0.00000073118280 , "800246" :0.00000055913978 , "104225" :0.00000053763441 , "131535" :0.00000053763441 , "130797" :0.00000043010753 , "127987" :0.00000006451613 , "100457" :0.00000005376344 , "172826" :0.00000005376344 , "130226" :0.00000004301075 , "659519" :0.00000002580645 , "130029" :0.00000001075269 }'
            ret = mr.tests_exec(forml, ret_str )  
            if ret_str: rett.append(ret)
            else:       rett.update(ret)
                

        return rett

        # for todo in self.todos:
        #     if todo['id'] == id:
        #         return todo
        # api.abort(404, "Todo {} doesn't exist".format(id))

    def create(self, data):        return "create"
    def update(self, id, data):    return "update"
    def delete(self, id):          return "delete"
DAO = fpDAO()

# _______________________
# ROUTES
# _______________________
@ns.route('/') # POST! 
class fpPredList(Resource):
    @ns.doc('get dummy')
    @ns.marshal_list_with(fp)
    def get(self):
        return {   "PRED": "25",  "FORML": "GET" }

    @ns.doc('post form and get fp')
    @ns.expect(fp)
    @ns.marshal_with(fp, code=201)
    def post(self):
        forml2  = api.payload["FORML"]
        forml2 = forml2.replace("'", '"'); print(forml2)
        ret_st  = api.payload["RET_STR"]
        ret_str = ( True  if ( ret_st == "X" ) else  False )
        # ret_str = "X"#False
        ret_str = False
        pred = DAO.get(forml2, ret_str)
        api.payload["FORML"] = "Done"

        if ret_str: 
            api.payload["PRED"] = pred
            api.payload["PRED_C2_0"] = pred[0]
            api.payload["PRED_C4_0"] = pred[1]
            api.payload["PRED_C1_0"] = pred[2]
        else: 
            api.payload["PRED"] = "SEPARATION! ret_str = False!"
            api.payload.update(pred)

        return api.payload
        # return DAO.create(api.payload), 201
        # EXAMPLE
        ex = {
               "FORML": "{ 'm':'1', '100023' : 1 }",
               "PRED" : "string"
             }

@api.route('/<string:forml>') # GET! 
# api.add_resource(Todo, '/todo/<int:todo_id>', endpoint='todo_ep')
# @api.route('/todo/<int:todo_id>', endpoint='todo_ep')
@ns.param('forml', 'Dummy par - not working!')
class fpPred(Resource):
    def get(self, forml):
        pred = 201
        if (len(request.form) > 0):
            forml2 = request.form['data'];  print(forml2)
            pred = DAO.get(forml2)
        # print(forml) # this is not working! 
        print(pred)
        return {"pred": pred}

    def put(self, forml):
        # forml = request.form['data']
        print(forml)
        return {"pred": 25}

# _______________________
# TESTS
# _______________________
def test_curl():
    pass
    # json_str = '''[
    #     { "m":"PBV10476AS", "178583" :0.74598 , "106104" :0.1 , "182789" :0.04 , "130172" :0.035 , "179661" :0.035 , "164421" :0.018 , "600040" :0.0108 , "116165" :0.008 , "164419" :0.0018 , "103396" :0.001 , "130217" :0.001 , "131460" :0.001 , "690750" :0.0007 , "611089" :0.0006 , "130058" :0.0004 , "130354" :0.0002 , "131101" :0.0002 , "131435" :0.00012 , "131136" :0.0001 , "131315" :0.0001 }   
    #     { "m":"1", "100023" : 1 }
    #     tmpLab = [50, 73]
    # $ curl http://localhost:5000/FORML{213} -d "data=Change my brakepads" -X GET
    # curl http://localhost:5000/form1Level -d "data='{ "m":"PBV10476AS", "178583" :0.74598 , "106104" :0.1 , "182789" :0.04 , "130172" :0.035 , "179661":0.035 , "164421" :0.018 , "600040" :0.0108 , "116165" :0.008 , "164419" :0.0018 , "103396" :0.001 , "130217" :0.001 , "131460" :0.001 , "690750" :0.0007 , "611089" :0.0006 , "130058" :0.0004 , "130354" :0.0002 , "131101" :0.0002 , "131435" :0.00012 , "131136" :0.0001 , "131315" :0.0001 }'" -X GET
    # curl http://localhost:5000/123 -d "data='{"m":"1","100023":1}'" -X GET
    #.curl http://localhost:5000/todo1

    # curl http://localhost:5000/123 -d "data='{'m':'1','100023':1}'" -X GET
    
    # curl http://localhost:5000/123 -d "data='
    # { 'm':'PBV10476AS', '178583' :0.74598 , '106104' :0.1 , '182789' :0.04 , '130172' :0.035 , '179661' :0.035 , '164421' :0.018 , '600040' :0.0108 , '116165' :0.008 , '164419' :0.0018 , '103396' :0.001 , '130217' :0.001 , '131460' :0.001 , '690750' :0.0007 , '611089' :0.0006 , '130058' :0.0004 , '130354' :0.0002 , '131101' :0.0002 , '131435' :0.00012 , '131136' :0.0001 , '131315' :0.0001 }   
    # '" -X GET

    # API.payload
    # {
    #   "hello": "{ 'm':'1', '100023' : 1 }",
    #   "pred": "string_real=73"  
    # }

def single_tests():
    """triple comment = treated as string """
    real = 50; #pred 68 , 67 -around 15%
    forml2 = '''{ "m":"PBV10476AS", "178583" :0.74598 , 
                                    "106104" :0.1 , 
                                    "182789" :0.04 , 
                                    "130172" :0.035 , 
                                    "179661" :0.035 , 
                                    "164421" :0.018 , 
                                    "600040" :0.0108 , 
                                    "116165" :0.008 , 
                                    "164419" :0.0018 ,
                                    "103396" :0.001 , 
                                    "130217" :0.001 , 
                                    "131460" :0.001 , 
                                    "690750" :0.0007 , 
                                    "611089" :0.0006 , 
                                    "130058" :0.0004 , 
                                    "130354" :0.0002 , 
                                    "131101" :0.0002 , 
                                    "131435" :0.00012 , 
                                    "131136" :0.0001 , 
                                    "131315" :0.0001 }  ''' 
    # real = 73 #pred 73 high, 77,74...
    # forml2 = '''{ "m":"1",          "100023" : 1 }    '''

    real = 54 #pred 57, 58 low prod
    forml2 = '''{ "m":"125660",     "160102" :0.31245 , 
                                    "131504" :0.1 , 
                                    "131030" :0.1 , 
                                    "100524" :0.07 , 
                                    "170036" :0.06 , 
                                    "130520" :0.06 , 
                                    "661269" :0.05 , 
                                    "106184" :0.035 , 
                                    "103249" :0.03 , 
                                    "690965" :0.02 , 
                                    "130255" :0.02 , 
                                    "104283" :0.02 , 
                                    "130141" :0.015 , 
                                    "130298" :0.015 , 
                                    "103828" :0.015 , 
                                    "606009" :0.01 , 
                                    "611073" :0.01 , 
                                    "611089" :0.01 , 
                                    "660554" :0.01 , 
                                    "600101" :0.008 , 
                                    "116165" :0.005 , 
                                    "611357" :0.005 , 
                                    "131487" :0.005 , 
                                    "638257" :0.003 , 
                                    "103798" :0.002 , 
                                    "103799" :0.002 , 
                                    "198039" :0.0015 , 
                                    "611025" :0.001 , 
                                    "131437" :0.001 , 
                                    "105169" :0.001 , 
                                    "130065" :0.001 , 
                                    "197635" :0.0005 , 
                                    "131395" :0.0005 , 
                                    "131080" :0.0005 , 
                                    "100202" :0.0005 , 
                                    "131091" :0.00005 } '''; 
    
    
    real = 101 #pred 101, 100
    forml2 = '''{ "m":"130935",     "130935" :1 } '''
    # forml2 = '''{ "m":"130935",     "342" :1 } ''' -> error test
    
    beginc = """
    real = 88  #pred 89,88
    forml2 = '''{ "m":"151436",     "164419" :0.531 , 
                                "718425" :0.2 ,   
                                "130520" :0.07 , 
                                "654060" :0.06 , 
                                "100751" :0.02 , 
                                "130412" :0.02 , 
                                "600105" :0.02 , 
                                "611105" :0.02 , 
                                "654040" :0.02 , 
                                "128715" :0.01 , 
                                "130879" :0.01 , 
                                "600194" :0.01 , 
                                "100476" :0.005 , 
                                "131002" :0.004 }   '''
    beginc = ""
    """ #end of comment
    
    pred = DAO.get(forml2)
    print("\n\n\n_R = {} and P = {}" .format(real ,pred ) )
    # print(md.dsp[["M","FP"]]);     # print(md.dsp.iloc[0])
    # md.print_form2(md.dsp.iloc[0])

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0", debug=True)  # accessible from the network! 
    # single_tests()   