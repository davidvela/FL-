from flask import Flask, request
from flask_restplus import Api, Resource, fields
from werkzeug.contrib.fixers import ProxyFix
from datetime import datetime
import mRun as mr
import utils_data as md


app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0', title='FP API',   description='Prototype to Predict FP API', )

ns = api.namespace('fp', description='FP Operations')

fp = api.model('fp', {
    #'id': fields.Integer(readOnly=True, description='The fp unique identifier'),
    'forml': fields.String(required=True, description='The form details'),
    'pred': fields.String(required= False,description='Prediction') })

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

class fpDAO(object):
    def __init__(self):
        self.counter = 0
        self.todos = []

    def get(self, forml ):
        print("___Start!___" +  datetime.now().strftime('%H:%M:%S')  )
        final = "_" ;  md.DESC = "FRALL1"; # FRAFLO / FRALL1
        execc = get_models(md.DESC)
        ex = execc[2]
        md.spn = ex["spn"]; md.dType = ex["dt"]; mr.epochs = ex["e"]; mr.lr = ex["lr"]; mr.h = ex["h"]
      
        mr.ninp = 1814
        mr.ninp, mr.nout, mr.top_k = md.getnn(mr.ninp)
        
        md.MODEL_DIR = md.LOGDIR + md.DESC + '/'   + mr.get_hpar(mr.epochs, final=final) +"/" 
        mr.model_path = md.MODEL_DIR + "model.ckpt" 
        mr.build_network3()                                                                                                                                                                                                                                                                                    
       
        print(mr.model_path)    
       
        ex["pe"]   = mr.evaluate( )
        ex["pt"] = mr.tests(url_test, p_col=False  )


        for todo in self.todos:
            if todo['id'] == id:
                return todo
        api.abort(404, "Todo {} doesn't exist".format(id))

    def create(self, data):
        return "create"
        # todo = data
        # todo['id'] = self.counter = self.counter + 1
        # self.todos.append(todo)
        # return todo

    def update(self, id, data):
        return "update"

    def delete(self, id):
        return "delete"


DAO = fpDAO()
# @ns.route('/')
# class TodoList(Resource):

@api.route('/<string:forml>')
# api.add_resource(Todo, '/todo/<int:todo_id>', endpoint='todo_ep')
# @api.route('/todo/<int:todo_id>', endpoint='todo_ep')
@ns.param('forml', 'Dummy par - not working!')
class fpPred(Resource):
    def get(self, forml):
        forml2 = '{ "m":"1", "100023" : 1 }'
        if (len(request.form) > 0):
            forml2 = request.form['data'];  print(forml2)
        
        print(forml) # this is not working! 
        
        # pred = DAO.get(forml2)
        return {"pred": 24}

    def put(self, forml):
        # forml = request.form['data']
        print(forml)
        return {"pred": 25}

        
if __name__ == '__main__':
    app.run(debug=True)


def test_curl():
    pass
    # { "m":"PBV10476AS", "178583" :0.74598 , "106104" :0.1 , "182789" :0.04 , "130172" :0.035 , "179661" :0.035 , "164421" :0.018 , "600040" :0.0108 , "116165" :0.008 , "164419" :0.0018 , "103396" :0.001 , "130217" :0.001 , "131460" :0.001 , "690750" :0.0007 , "611089" :0.0006 , "130058" :0.0004 , "130354" :0.0002 , "131101" :0.0002 , "131435" :0.00012 , "131136" :0.0001 , "131315" :0.0001 }   
    # { "m":"1", "100023" : 1 }
    # $ curl http://localhost:5000/FORML{213} -d "data=Change my brakepads" -X GET
    # curl http://localhost:5000/form1Level -d "data='{ "m":"PBV10476AS", "178583" :0.74598 , "106104" :0.1 , "182789" :0.04 , "130172" :0.035 , "179661":0.035 , "164421" :0.018 , "600040" :0.0108 , "116165" :0.008 , "164419" :0.0018 , "103396" :0.001 , "130217" :0.001 , "131460" :0.001 , "690750" :0.0007 , "611089" :0.0006 , "130058" :0.0004 , "130354" :0.0002 , "131101" :0.0002 , "131435" :0.00012 , "131136" :0.0001 , "131315" :0.0001 }'" -X GET
    #.