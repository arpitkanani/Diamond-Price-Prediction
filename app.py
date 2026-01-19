
from flask import Flask,request,render_template,Response,redirect, url_for, session
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
import warnings 
warnings.filterwarnings('ignore')


application=Flask(__name__)
app=application
app.secret_key = "gemstone_secret_key"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoints():
    result = None
    if request.method == 'POST':
        data = CustomData(
            cut=str(request.form.get('cut')),
            color=str(request.form.get('color')),
            clarity=str(request.form.get('clarity')),
            carat=float(request.form.get('carat') or 0),
                depth=float(request.form.get('depth') or 0),
                table=float(request.form.get('table') or 0),
                x=float(request.form.get('x') or 0),
                y=float(request.form.get('y') or 0),
                z=float(request.form.get('z') or 0)
        )

        pred_data = data.get_data_as_dataframe()
        prediction_pipeline = PredictPipeline()

        prediction = prediction_pipeline.predict(pred_data)
        #result = round(float(prediction[0]), 2)   

        result = round(float(prediction[0]), 2)
        
    
        #result = session.pop('prediction_result', None)
        #return render_template('home.html', result=result)
    return render_template('home.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True) 