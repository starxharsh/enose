import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')

class TraditionalModel:
    def __init__(self, model, param_grid, task='classification', is_multi_output=False):
        self.model = model
        self.param_grid = param_grid
        self.task = task
        self.is_multi_output = is_multi_output
        self.best_model = None
        self.best_params = None

    def train(self, X_train, y_train):
        if self.is_multi_output and self.task == 'classification':
            if not isinstance(self.model, (RandomForestClassifier, XGBClassifier)):
                self.model = MultiOutputClassifier(self.model)
                self.param_grid = {f'estimator__{k}': v for k, v in self.param_grid.items()}

        grid_search = GridSearchCV(self.model, self.param_grid, cv=5, n_jobs=-1, scoring='accuracy' if self.task == 'classification' else 'neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        return self.best_model

    def predict(self, X_test):
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        return self.best_model.predict(X_test)

    def get_best_params(self):
        return self.best_params

def get_knn(task='classification', is_multi_output=False):
    model = KNeighborsClassifier() if task == 'classification' else KNeighborsRegressor()
    param_grid = {'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
    return TraditionalModel(model, param_grid, task, is_multi_output)

def get_svm(task='classification', is_multi_output=False):
    model = SVC(probability=True) if task == 'classification' else SVR()
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']}
    return TraditionalModel(model, param_grid, task, is_multi_output)

def get_rf(task='classification', is_multi_output=False):
    model = RandomForestClassifier(random_state=42) if task == 'classification' else RandomForestRegressor(random_state=42)
    param_grid = {'n_estimators': [100, 300, 500], 'max_depth': [10, 20, None]}
    return TraditionalModel(model, param_grid, task, is_multi_output)

def get_xgboost(task='classification', is_multi_output=False):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42) if task == 'classification' else XGBRegressor(random_state=42)
    param_grid = {'learning_rate': [0.01, 0.1, 0.3], 'max_depth': [3, 6, 9], 'n_estimators': [100, 300]}
    return TraditionalModel(model, param_grid, task, is_multi_output)

def run_all_traditional_models(X_train, X_test, y_train, y_test, task='classification'):
    is_multi_output = False
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        is_multi_output = True

    models = {
        'KNN': get_knn(task, is_multi_output),
        'SVM': get_svm(task, is_multi_output),
        'Random Forest': get_rf(task, is_multi_output),
        'XGBoost': get_xgboost(task, is_multi_output)
    }

    if task == 'classification':
        lda_model = TraditionalModel(LinearDiscriminantAnalysis(), {'solver': ['svd', 'lsqr', 'eigen']}, task, is_multi_output)
        models['LDA'] = lda_model

    results = {}
    for name, wrapper in models.items():
        try:
            print(f"Training {name}...")
            wrapper.train(X_train, y_train)
            predictions = wrapper.predict(X_test)
            results[name] = {
                'best_params': wrapper.get_best_params(),
                'predictions': predictions,
                'model': wrapper.best_model
            }
        except Exception as e:
            print(f"Failed to train {name}: {e}")

    if task == 'classification':
        print("Running KMeans...")
        kmeans_results = {}
        for k in [4, 8, 16]:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_train)
            if len(set(labels)) > 1:
                silhouette = silhouette_score(X_train, labels)
                ari = -1
                if not is_multi_output:
                    ari = adjusted_rand_score(y_train, labels)
                kmeans_results[k] = {'silhouette_score': silhouette, 'adjusted_rand_score': ari}
        results['KMeans'] = kmeans_results

    return results
