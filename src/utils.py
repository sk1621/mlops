import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluation_model(X_train,y_train,X_test,y_test,models,params):
    report = {}
    try:
        for i in range(len(list(models))):
            model = list(models.values()[i])
            param = params[list(models.keys())[i]]
            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model = fit(X_train,y_train)

            y_train_predicted = model.predict(X_train)
            y_test_predicted = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_predicted)
            test_model_score = r2_score(y_test,y_test_predicted)
            report[list(models.keys()[i])] = test_model_score

        return report,train_model_score
        
    except Exception as e:
        raise CustomException(e,sys)