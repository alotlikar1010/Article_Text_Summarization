from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin

from Text_Summ.pipeline.train import TrainingPipeline
from Text_Summ.pipeline.prediction_pipeline import SinglePrediction

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/train',methods=['POST'])
@cross_origin()
def train():
    train_pipeline = TrainingPipeline()
    train_pipeline.run_pipeline()
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
@cross_origin()
def getsummary():
    if request.method == 'GET':
        return render_template('summary.html')
    else:
        
        input_text = request.form['data']
        single_prediction = SinglePrediction()
        result = single_prediction.predict(input_text)
        return render_template('summary.html',result=result)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)