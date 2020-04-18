#coding: utf-8
import pandas as pd
from datetime import datetime

class RegressionModel:

    linearRegression_1 = None   # instanceof sklearn.LinearModel.LinearRegression
    linearRegression_2 = None   # instanceof sklearn.LinearModel.LinearRegression
    polynomialFeatures = None   # instanceof sklearn.preprocessing.PolynomialFeatures
    firebase = None             # instanceof firebase.FirebaseApplication

    def __init__(self, firebaseUrl, method, targetAtribute, independentVariables):
        self.firebaseUrl = firebaseUrl
        self.method = method
        self.targetAtribute = targetAtribute
        self.independentVariables = independentVariables                                                         

    def getFirebaseUrl(self):
        return self.firebaseUrl

    def setFirebaseUrl(self, firebaseUrl=None):
        self.firebaseUrl = firebaseUrl

    def getDatabase(self):
        return self.firebase

    def getMethod(self):
        return self.method

    def convertDateTime(self, database=None, dateArgs=None, hourArgs=None, separator='-'):
        '''
        Transforma os os campos data e hora em objetos do tipo Datetime
        '''
        X = []
        for a in database.values():
            string = str(a[''+dateArgs])+separator+str(a[''+hourArgs])
            value = string.split(separator)
            x = value[3].split(':')
            data = datetime(int(value[0]), int(value[1]), int(
                 value[2]), int(x[0]), int(x[1]), 00)
            X.append(data.timestamp())
        print(X)
        return X

    def separatingNumericData(self, database=None):
        ds = []        # dados da raiz 'sensores'
        dfOpen = None  # DataFrame de entrada
        dfExit = None  # DataFrame de saida

        for a in database.values():
            ds.append(a['sensores'])

        dfOpen = pd.DataFrame(ds[0])

        for a in range(1, len(ds)):
            aux = pd.DataFrame(ds[a])
            dfOpen = pd.concat([dfOpen, aux])

        dfOpen.set_index('tipo', inplace=True)
        dfExit = pd.DataFrame(list(dfOpen['dado']['' + self.targetAtribute]), columns=['' + self.targetAtribute])
        for a in self.independentVariables:
            dfExit['' + a] = list(dfOpen['dado']['' + a])    

        print(dfExit)
        return dfExit

    def selectAttributes(self, database=None):
        x = None  # variáveis independentes
        y = None  # atributo alvo

        y = database[''+self.targetAtribute]
        y = pd.DataFrame(y, columns=[''+self.targetAtribute])
        print(y)

        if type(self.independentVariables) == 'list':
            x = database[self.independentVariables]
        x = database[self.independentVariables]
        print(x)

        return y, x

