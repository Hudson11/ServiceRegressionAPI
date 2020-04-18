from flask import Flask, request, Response
from flask_restful import Api
from flask_jsonpify import jsonify
from ModelsRegression.LinearRegression import LinearRegression_
from ModelsRegression.SvrRegression import SvrRegression
from ModelsRegression.MlpRegression import MlpRegression
from datetime import datetime

#Construtor Flask
app = Flask(__name__)
api = Api(app)

linearRegression = None        # Object ModelsRegression.LinearDao.LinearDao
svrRegression = None           # Object ModelsRegression.Svrdao.SvrDao
mlpRegression = None           # Object ModelsRegression.Mlpdao.MlpDao

@app.route('/prediction/linearRegression', methods=['POST'])
def trainingModelLinear():

    global linearRegression

    if request.method == 'POST':

        body = request.get_json()
        try:
            linearRegression = LinearRegression_(body['firebaseUrl'], body['targetAttribute'], body['idNode'], body['independentVariables'])
        except:
            return jsonify({'error': 'argumentError'})

        linearRegression.training()
        #Obetem os valores de avaliação do modelo
        result = linearRegression.modelEvaluation()

        mean = linearRegression.getMeanTargetAttribute(targetAttribute=body['targetAttribute'])

        return jsonify(
                        {
                            'response':{
                                'method':'LinearRegression', 
                                'result':'200 OK',
                                'meanSquaredError': result[1],
                                'meanAbsoluteError': result[0],
                                'meanAttribute': float(mean)
                            },
                        }
                    )

    else:

        try:
            return jsonify({'data':linearRegression.linearEquation([98, 77, 1558384740.0])})
        except:
            return jsonify({'error': 'requires training'})

@app.route('/prediction/linearRegression/data', methods=['POST'])
def consultModelLinear():

    global linearRegression
    time = None

    body = request.get_json()

    try:
        if body['time'] is not None:
            value = str(body['time']).split('-')
            date = value[0].split('/')
            hours = value[1].split(':')
            time = datetime(int(date[0]), int(date[1]), int(date[2]), int(hours[0]), int(hours[1]), int(hours[2]))
    except KeyError:
        time = datetime.now()

    try:
        return jsonify({'data_predict': linearRegression.linearEquation([body['arguments'][0], body['arguments'][1], time.timestamp()])})
    except AttributeError:
        return jsonify({'error': 'requires training'})
    except KeyError:
        return jsonify({'KeyError': 'error arguments'})

@app.route('/prediction/svrRegression', methods=['POST'])
def trainingModelSvr():

    global svrRegression

    if request.method == 'POST':

        body = request.get_json()

        try:
            svrRegression = SvrRegression(body['firebaseUrl'], body['targetAttribute'], body['idNode'], body['independentVariables'])
        except:
            return jsonify({'error': 'argumentError'})

        svrRegression.training()
        #Obetem os valores de avaliação do modelo
        result = svrRegression.modelEvaluation()

        mean = svrRegression.getMeanTargetAttribute(targetAttribute=body['targetAttribute'])
        print(mean)

        return jsonify(
                        {
                            'response':{
                                'method':'SvrRegression', 
                                'result':'ok',
                                'meanSquaredError': result[1],
                                'meanAbsoluteError': result[0],
                                'meanAttribute': float(mean)
                            }
                        }
                    )

    else:

        try:
            return jsonify({'data':svrRegression.srvEquation([98, 77, 1558384740.0])})
        except:
            return jsonify({'error': 'requires training'})

@app.route('/prediction/svrRegression/data', methods=['POST'])
def consultModelSvr():

    global svrRegression
    time = None

    body = request.get_json()

    try:
        if body['time'] is not None:
            value = str(body['time']).split('-')
            date = value[0].split('/')
            hours = value[1].split(':')
            time = datetime(int(date[0]), int(date[1]), int(date[2]), int(hours[0]), int(hours[1]), int(hours[2]))
    except KeyError:
        time = datetime.now()

    try:
        return jsonify({'data_predict': svrRegression.srvEquation([body['arguments'][0], body['arguments'][1], time.timestamp()])})
    except AttributeError:
        return jsonify({'error': 'requires training'})
    except KeyError:
        return jsonify({'KeyError': 'error arguments'})


@app.route('/prediction/mlpRegression', methods=['POST'])
def trainigModelMlp():

    global mlpRegression

    if request.method == 'POST':

        body = request.get_json()

        try:
            mlpRegression = MlpRegression(body['firebaseUrl'], body['targetAttribute'], body['idNode'], body['independentVariables'])
        except:
            return jsonify({{'error': 'argumentError'}})

        mlpRegression.training()
        #Obetem os valores de avaliação do modelo
        result = mlpRegression.modelEvaluation()

        mean = linearRegression.getMeanTargetAttribute(targetAttribute=body['targetAttribute'])

        return jsonify(
                        {
                            'response':{
                                'method':'MlpRegression', 
                                'result':'ok',
                                'meanSquaredError': result[1],
                                'meanAbsoluteError': result[0],
                                "meanAttribute": float(mean)
                            }
                        }
                    )

    else:

        try:
            return jsonify({'data':mlpRegression.mplEquation([98, 77, 1558384740.0])})
        except:
            return jsonify({'error': 'requires training'})

@app.route('/prediction/mlpRegression/data', methods=['POST'])
def consultModelMlp():

    global mlpRegression
    time = None

    body = request.get_json()

    try:
        if body['time'] is not None:
            value = str(body['time']).split('-')
            date = value[0].split('/')
            hours = value[1].split(':')
            time = datetime(int(date[0]), int(date[1]), int(date[2]), int(hours[0]), int(hours[1]), int(hours[2]))
    except KeyError:
        time = datetime.now()

    try:
        return jsonify({'data_predict': mlpRegression.mlpEquation([body['arguments'][0], body['arguments'][1], time.timestamp()])})
    except AttributeError:
        return jsonify({'error': 'requires training'})
    except KeyError:
        return jsonify({'KeyError': 'error arguments'})

if __name__ == '__main__':
    app.run(debug=True)
