from flask import *
import pandas as pd
from requests import put, get
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class CalcForm(FlaskForm):
    forml = StringField('forml', validators=[DataRequired()])
    submit = SubmitField('Calc')

app = Flask(__name__)
app.config.update(dict(
    SECRET_KEY="powerful secretkey",
    WTF_CSRF_SECRET_KEY="a csrf secret key"
))
app.config['SERVER_NAME'] = '127.0.0.1' + ':' + '5002'
# flask_app.config['SERVER_NAME'] = server_name + ':' + server_port
# flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = settings.RESTPLUS_SWAGGER_UI_DOC_EXPANSION # 'list'
# flask_app.config['RESTPLUS_VALIDATE'] = settings.RESTPLUS_VALIDATE # True
# flask_app.config['RESTPLUS_MASK_SWAGGER'] = settings.RESTPLUS_MASK_SWAGGER # False
# flask_app.config['ERROR_404_HELP'] = settings.RESTPLUS_ERROR_404_HELP # False 
#                   - If a request does not match any of the application endpoints => return error 404 or not 

user = {'username': 'David_A'}

@app.route('/')
@app.route('/index')
def index():
    form = CalcForm( )
    form.forml.data = "hola1"


    return render_template('index.html', title='Home', user=user, form = form)

    # print(get('http://localhost:5000/hello').json())
    # return "Hello, World!"
    # ret = get('http://localhost:5000/hello').json()
    # print(ret)
    # return "hello__" + ret["hello"] #"Hello, World!"
    
@app.route('/Calc', methods=['GET', 'POST'])
def Calc():
    # return render_template('index.html', title='Home', user=user )
    # form = CalcForm()
    form = CalcForm(request.form)
    # print(form.forml)

    # if form.validate_on_submit():  # press button
    if request.method == 'POST': # and form.validate():
        # CALCULATION LOGIC: 
        # ret = get('http://localhost:5000/dummy', data={'data': form.forml.data }).json()
        # pred = ret ... 
        # + error handling 200, 201 ... 404 codes! 
        # print(ret)        
        
        pred = 101
        flash('Calc requested for form {}'.format( form.forml.data))
        flash('pred={}'.format( pred ))
        return render_template('index.html', title='Home', user=user, form = form  )

        # return redirect(url_for('index'))
        # return redirect('/index2')
        # return redirect(url_for('index'))
    form.forml.data = "hola2"
    flash('Not preshed!')
    return render_template('index.html', title='Home', user=user, form = form )



if __name__ == "__main__":
    app.run(debug=True)  

def test_curl():
    pass
    # test data
    # json_str = '''[
    #     { "m":"PBV10476AS", "178583" :0.74598 , "106104" :0.1 , "182789" :0.04 , "130172" :0.035 , "179661" :0.035 , "164421" :0.018 , "600040" :0.0108 , "116165" :0.008 , "164419" :0.0018 , "103396" :0.001 , "130217" :0.001 , "131460" :0.001 , "690750" :0.0007 , "611089" :0.0006 , "130058" :0.0004 , "130354" :0.0002 , "131101" :0.0002 , "131435" :0.00012 , "131136" :0.0001 , "131315" :0.0001 }   
    #     { "m":"1", "100023" : 1 }
    #     tmpLab = [50, 73]
    idea = 8 # Better separation.
    # { "m":"1", "100023" : 1 }{ "m":"1", "100023" : 1 }{ "m":"1", "100023" : 1 }{ "m":"1", "100023" : 1 }{ "m":"1", "100023" : 1 }{ "m":"1", "100023" : 1 }{ "m":"1", "100023" : 1 }
