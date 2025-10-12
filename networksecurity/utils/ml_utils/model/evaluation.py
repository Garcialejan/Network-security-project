import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from networksecurity.exception.exception import NetworkSecurityException

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}
        best_params_per_model = {}

        for model_name, model in models.items():
            para = param.get(model_name, {})

            if para:
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)
                best_params = gs.best_params_
                model.set_params(**best_params)
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
                best_params = {}

            y_test_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            best_params_per_model[model_name] = best_params

        return report, best_params_per_model

    except Exception as e:
        raise NetworkSecurityException(e, sys)