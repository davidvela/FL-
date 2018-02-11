from flask import Flask
from flask_restplus import Api, Resource, fields
from werkzeug.contrib.fixers import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0', title='FP API',   description='Prototype to Predict FP API', )

ns = api.namespace('fp', description='FP Operations')

todo = api.model('fp', {
    # 'id': fields.Integer(readOnly=True, description='The fp unique identifier'),
    'form': fields.String(required=True, description='The form details'),
    'pred': fields.String(required= False,description='Prediction') ) })

class fpDAO(object):
    def __init__(self):
        self.counter = 0
        self.todos = []

    def get(self, id):
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
        # todo = self.get(id)
        # todo.update(data)
        # return todo

    def delete(self, id):
        return "delete"
        # todo = self.get(id)
        # self.todos.remove(todo)

DAO = fpDAO()
# @ns.route('/')
# class TodoList(Resource):

@api.route('/<string:todo_id>')
# api.add_resource(Todo, '/todo/<int:todo_id>', endpoint='todo_ep')
# @api.route('/todo/<int:todo_id>', endpoint='todo_ep')
class TodoSimple(Resource):
    def get(self, todo_id):
        return {todo_id: todos[todo_id]}

    def put(self, todo_id):
        todos[todo_id] = request.form['data']
        return {todo_id: todos[todo_id]}

        
if __name__ == '__main__':
    app.run(debug=True)