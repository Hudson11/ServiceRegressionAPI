#coding: utf-8
from Modelo.RegressionModel import RegressionModel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import pandas as pd
from firebase import firebase

class LinearRegression_:

    linearRegression = None
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

        self.linearModel(x=self.x_train, y=self.y_train)

    def linearModel(self, x=None, y=None):
        '''
        :param x: Parametros de independentes
        :param y: Atributo Alvo
        :return: representacao compacta dos dados (Equacao de regressao)
        '''
        print('Linear Regression Activate')
        self.linearRegression = LinearRegression()  # Instance Object
        self.linearRegression.fit(self.x_train, self.y_train)

    # Retorna um Valor de Predição
    def linearEquation(self, value):
        '''
            Função usada para predição de um dado específico, usada somente se o modelo linear (function linearModel())
            estiver sido preparado, retorna um valor X estimado.
            :return: Retorna o valor estimada de y de acordo com os parâmetros x´s passados
        '''
        return float(self.linearRegression.predict([value]))

    def modelEvaluation(self):
        '''
        :return: List com resultados da avaliação (Média do Erro absoluto e a raiz quadrada
            do erro).
        '''
        result = self.linearRegression.predict(self.x_test)
        data = []
        data.append(metrics.mean_absolute_error(result, self.y_test))
        data.append(metrics.mean_squared_error(result, self.y_test))
        return data

    def getIdNode(self):
        return self.__idNode

    def getFirebase(self):
        return self.__firebase

    def getMeanTargetAttribute(self, targetAttribute=None):
        database = self.firebase.get('/Nodes/' + str(self.getIdNode()), None)
        df = self.regressionModel.separatingNumericData(database=database)
        df = pd.DataFrame(df['' + str(targetAttribute)])
        return df.median()
