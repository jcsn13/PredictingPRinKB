from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.GwasFeatureExtractor import GwasFeatureExtractor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd


# ################################################## #
# Author: Jose Claudio Soares                        #
# Email: joseclaudiosoaresneto@gami.com              #
# Date(mm/dd/yyyy): 11/03/2020                       #
# Github: jcsn13                                     #
# Credits: github.com/crowegian/AMR_ML               #
# ################################################## #


class PredictPRinKP:
    def __init__(self, datasets_full=None, datasets_gwas=None):
        self.dict_dataframes = {}
        self.dataset_full = datasets_full
        self.datasets_gwas = datasets_gwas
        self.random_state = np.random.RandomState(1)

    def read_datasets(self):
        if self.dataset_full:
            for i in self.dataset_full:
                df_full = pd.read_csv('../data/' + i + 'full.csv')
                df_full.set_index('isolate', drop=True, inplace=True)

                pb_res = df_full['pbr_res']
                df_full = df_full.drop(columns=['pbr_res', 'mgrB'], axis=1)

                self.dict_dataframes['pbr_res_' + i] = pb_res
                self.dict_dataframes['df_full_' + i] = df_full

        if self.datasets_gwas:
            for i in self.datasets_gwas:
                df_train = pd.read_csv('../data/' + i + 'train.csv')

                df_test = pd.read_csv('../data/' + i + 'test.csv')

                df_gwas = pd.read_csv('../data/' + i + 'gwas.csv')
                df_gwas.set_index('LOCUS_TAG')

                df_full = pd.concat([df_test, df_train])

                pb_res = df_full[['pbr_res']]

                df_full = df_full.drop(columns=['pbr_res'], axis=1)

                self.dict_dataframes['pbr_res_' + i] = pb_res
                self.dict_dataframes['df_full_' + i] = df_full
                self.dict_dataframes['df_gwas_' + i] = df_gwas

    @staticmethod
    def perform_train_test_split(dataset, metadata):
        if len(dataset) == 1:
            x_train, x_test, y_train, y_test = train_test_split(dataset[0].values, metadata[0].values, test_size=0.25,
                                                                random_state=42)
        else:
            gwas_feature_extractor = GwasFeatureExtractor(x_locus_tags=dataset[1]['LOCUS_TAG'], gwas_df=dataset[1])
            gwas_feature_extractor.fit()
            x = gwas_feature_extractor.transform(dataset[0])
            x = x.drop('isolate', axis=1)

            x_train, x_test, y_train, y_test = train_test_split(x.values, metadata[0].values.ravel(), test_size=0.25,
                                                                random_state=42)

        return x_train, x_test, y_train, y_test

    @staticmethod
    def print_performance_metrics(grid, name, x_test, y_test):
        print("\n*********{}*********".format(name))
        print(
            "Performed GridSearchCV 10\nBest CV params {}\nBest CV accuracy {}"
            "\nTest accuracy of best grid search hypers:"
            .format(grid.best_params_, grid.best_score_), grid.score(x_test, y_test))

    def perform_train_full(self):
        print('\n############################# Full datasets #############################')
        for i in self.dataset_full:
            print('----------------Training {}----------------'.format(i))
            dataset = [self.dict_dataframes['df_full_' + i]]
            metadata = [self.dict_dataframes['pbr_res_' + i]]
            name = i
            self.train_SGDClassifier(dataset, metadata, name)
            self.train_logistic_regression(dataset, metadata, name)
            self.train_SVC(dataset, metadata, name)
            self.train_random_forest(dataset, metadata, name)
            self.train_GBTC(dataset, metadata, name)
        return self

    def perform_train_gwas(self):
        print('\n############################# GWAS datasets #############################')
        for i in self.datasets_gwas:
            print('----------------Training {}----------------'.format(i))
            dataset = [self.dict_dataframes['df_full_' + i],
                       self.dict_dataframes['df_gwas_' + i]]
            metadata = [self.dict_dataframes['pbr_res_' + i]]
            name = i
            self.train_SGDClassifier(dataset, metadata, name)
            self.train_logistic_regression(dataset, metadata, name)
            self.train_SVC(dataset, metadata, name)
            self.train_random_forest(dataset, metadata, name)
            self.train_GBTC(dataset, metadata, name)
        return self

    def train_SGDClassifier(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = self.perform_train_test_split(dataset, metadata)

        parameters = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                      'loss': ['hinge', 'log'], 'penalty': ['l1', 'l2', 'elasticnet'],
                      'n_jobs': [-1]}
        model = SGDClassifier(random_state=self.random_state, max_iter=10000)
        grid = GridSearchCV(model, parameters, cv=10)
        grid.fit(x_train, y_train)

        model_info = ["Stochastic Gradient Descent Classifier", x_test, y_test, "SGDC"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2])

        return self

    def train_logistic_regression(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = self.perform_train_test_split(dataset, metadata)

        parameters = {'C': [0.1, 1, 10, 100], 'penalty': ['l2'],
                      'multi_class': ['auto', 'multinomial']}
        model = LogisticRegression(random_state=self.random_state, max_iter=10000)
        grid = GridSearchCV(model, parameters, cv=10)
        grid.fit(x_train, y_train)

        model_info = ["Logistic Regression", x_test, y_test, "LR"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2])

        return self

    def train_SVC(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = self.perform_train_test_split(dataset, metadata)

        parameters = {'C': [0.1, 1, 10, 100], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                      'kernel': ['rbf', 'poly', 'sigmoid']}
        model = SVC(random_state=self.random_state, max_iter=10000)
        grid = GridSearchCV(model, parameters, cv=10)
        grid.fit(x_train, y_train)

        model_info = ["Support Vectors Classifier", x_test, y_test, "SVC"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2])

        return self

    def train_GBTC(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = self.perform_train_test_split(dataset, metadata)

        parameters = {'learning_rate': [0.001, 0.01, 0.1], 'n_estimators': [50, 100, 200, 300, 400, 500],
                      'max_depth': [1, 3, 5, 10, 12]}
        model = GradientBoostingClassifier(random_state=self.random_state)
        grid = GridSearchCV(model, parameters, cv=10)
        grid.fit(x_train, y_train)

        model_info = ["Gradient Boosting Classifier", x_test, y_test, "GBTC"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2])
        return self

    def train_random_forest(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = self.perform_train_test_split(dataset, metadata)

        parameters = {'n_estimators': [5, 10, 15, 20], 'max_features': ["sqrt", "log2", 0.5],
                      'max_depth': [None], 'criterion': ['gini', 'entropy']}
        model = RandomForestClassifier(random_state=self.random_state)
        grid = GridSearchCV(model, parameters, cv=10)
        grid.fit(x_train, y_train)

        model_info = ["Random Forest", x_test, y_test, "RF"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2])

        return self

    def main(self):
        self.read_datasets()

        if self.dataset_full:
            self.perform_train_full()
        if self.datasets_gwas:
            self.perform_train_gwas()


if __name__ == "__main__":
    # stdoutOrigin = sys.stdout
    # sys.stdout = open("log.txt", "w")

    df01 = ['dataset_1_', 'dataset_3_']
    df02 = ['dataset_19_', 'dataset_20_', 'dataset_21_']

    PR = PredictPRinKP(datasets_full=df01, datasets_gwas=df02)
    PR.main()

    # sys.stdout.close()
    # sys.stdout = stdoutOrigin

    print("finished model training")
