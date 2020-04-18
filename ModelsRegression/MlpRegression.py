#coding: utf-8
from Modelo.RegressionModel import RegressionModel
from firebase import firebase
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.neural_network import MLPRegressor
import pandas as pd

class MlpRegression:

    mlpRegression = None
    regressionModel = None
    firebase = None

    x_test = None
    y_test = None

    x_train = None
    x_test = None

    def __init__(self, firebaseUrl, targetAttribute, idNode, independentVariables):
        self._firebaseUrl = firebaseUrl
        self._targetAttribute = targetAttribute
        self._idNode = idNode
        self._independentVariables = independentVariables

        if firebaseUrl != None:
            self.regressionModel = RegressionModel(firebaseUrl, None, targetAttribute, independentVariables)
            self.firebase = firebase.FirebaseApplication(self.regressionModel.getFirebaseUrl(), None)

    def training(self):
        database = self.firebase.get('/Nodes/' + str(self.getIdNode()), None)

        df = self.regressionModel.separatingNumericData(database=database)
        y, x = self.regressionModel.selectAttributes(database=df)
        datetime = self.regressionModel.convertDateTime(database=database, dateArgs='data', hourArgs='hora')
        x['tempo'] = datetime

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.4)

        self.mlpModel(x=self.x_train, y=self.y_train)

    def mlpModel(self, x=None, y=None):
        '''
        :param x: Variáveis independetes da base
        :param y: Atributo alvo -> valor que queremos estimar
        :best_parans:
            {
                'activation': 'relu',
                'hidden_layer_sizes': (6,),
                'learning_rate': 'constant',
                'learning_rate_init': 0.1,
                'max_iter': 500,
                'momentum': 0.5,
                'solver': 'lbfgs'
            }
        '''
        self.mlpRegression = MLPRegressor(activation='relu', hidden_layer_sizes=(6,), learning_rate='constant', learning_rate_init=0.1
                                          , max_iter=500, momentum=0.5, solver='lbfgs')
        self.mlpRegression.fit(x, y)

    def mplEquation(self, value):
        '''
        :param value: Entrada com as informações independentes para estimar o atributo y
        :return: valor estimado para o atributoo y
        '''
        return float(self.mlpRegression.predict([value]))
    
    def modelEvaluation(self):
        '''
        :return: List com resultados da avaliação (Média do Erro absoluto e a raiz quadrada
            do erro).
        '''
        result = self.mlpRegression.predict(self.x_test)
        data = []
        data.append(metrics.mean_absolute_error(result, self.y_test))
        data.append(metrics.mean_squared_error(result, self.y_test))
        return data

    def getIdNode(self):
        return self._idNode

    def getMeanTargetAttribute(self, targetAttribute=None):
        database = self.firebase.get('/Nodes/' + str(self.getIdNode()), None)
        df = self.regressionModel.separatingNumericData(database=database)
        df = pd.DataFrame(df['' + str(targetAttribute)])
        return df.median()