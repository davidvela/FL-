from flask import Flask
from flask_restplus import Api, Resource, fields
from werkzeug.contrib.fixers import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0', title='FP API',   description='Prototype to Predict FP API', )

ns = api.namespace('fp', description='FP Operations')

todo = api.model('fp', {
    'id': fields.Integer(readOnly=True, description='The fp unique identifier'),
    'form': fields.String(required=True, description='The form details'),
    'pred': fields.String(required= False,description='Prediction') ) })
