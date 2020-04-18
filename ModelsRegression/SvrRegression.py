#coding: utf-8
from Modelo.RegressionModel import RegressionModel
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from firebase import firebase
import  pandas as pd

class SvrRegression:

    srvRegression = None
    regressionModel = None
    firebase = None

    x_test = None
    y_test = None

    x_train = None
    x_test = None

    def __init__(self, firebaseUrl, targetAttribute, idNode, independentVariables):
        self.__firebaseUrl = firebaseUrl
        self.__targetAttribute = targetAttribute
        self.__idNode = idNode
        self.__independentVariables = independentVariables

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

        self.srvModel(x=self.x_train, y=self.y_train)

    def srvModel(self, x=None, y=None):
        '''
            MELHOR PARAMETRO - 1º RODADA DE TESTES
            {
                'C': 11,
                'degree': 3,
                'kernel': 'sigmoid',
                'max_iter': 101
            }
        '''
        self.srvRegression = SVR(C=11, kernel='sigmoid', degree=3, max_iter=101)
        self.srvRegression.fit(x, y)

    def srvEquation(self, value):
        return float(self.srvRegression.predict([value]))
    
    def modelEvaluation(self):
        '''
        :return: List com resultados da avaliação (Média do Erro absoluto e a raiz quadrada
            do erro).
        '''
        result = self.srvRegression.predict(self.x_test)
        data = []
        data.append(metrics.mean_absolute_error(result, self.y_test))
        data.append(metrics.mean_squared_error(result, self.y_test))
        return data

    def getIdNode(self):
        return self.__idNode

    def getMeanTargetAttribute(self, targetAttribute=None):
        database = self.firebase.get('/Nodes/' + str(self.getIdNode()), None)
        df = self.regressionModel.separatingNumericData(database=database)
        df = pd.DataFrame(df['' + str(targetAttribute)])
        return df.median()