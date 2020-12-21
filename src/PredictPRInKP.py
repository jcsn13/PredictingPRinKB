from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
from src.GwasFeatureExtractor import GwasFeatureExtractor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_roc_curve, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import sys
import random
import pandas as pd
import scipy.stats


# ################################################## #
# Author: Jose Claudio Soares                        #
# Email: joseclaudiosoaresneto@gmail.com             #
# Date(mm/dd/yyyy): 11/03/2020                       #
# Github: jcsn13                                     #
# Credits: github.com/crowegian/AMR_ML               #
# ################################################## #


class PredictPRinKP:
    def __init__(self, datasets_full=None, datasets_gwas=None):
        self.dict_dataframes = {}
        self.dataset_full = datasets_full
        self.datasets_gwas = datasets_gwas
        self.SEED = np.random.RandomState(42)

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
                                                                random_state=42, shuffle=True)
        else:
            gwas_feature_extractor = GwasFeatureExtractor(x_locus_tags=dataset[1]['LOCUS_TAG'], gwas_df=dataset[1])
            gwas_feature_extractor.fit()
            x = gwas_feature_extractor.transform(dataset[0])
            # x = x.drop(['isolate'], axis=1)

            x_train, x_test, y_train, y_test = train_test_split(x.values, metadata[0].values.ravel(), test_size=0.25,
                                                                random_state=42, shuffle=True)

        return x_train, x_test, y_train, y_test

    @staticmethod
    def mean_confidence_interval(grid, x_test, y_test, confidence=0.96):
        score_array = []
        for i in range(200):
            x_sample_array = []
            y_sample_array = []

            range_ = int(len(y_test) * 0.65)

            for j in range(range_):
                index = np.random.randint(0, len(x_test) - 1)
                x = x_test[index]
                y = y_test[index]
                x_sample_array.append(x)
                y_sample_array.append(y)

            y_pred = grid.predict(x_sample_array)

            score = accuracy_score(y_pred, y_sample_array)
            score_array.append(score)

        a = 1.0 * np.array(score_array)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, h

    def print_performance_metrics(self, grid, name, x_test, y_test, dataset, short_name):
        print("\n*********{}*********".format(name))
        print(
            "Performed GridSearchCV 10\nBest CV params {}\nBest CV accuracy {}"
            "\nTest accuracy(confidence interval 0.95) of best grid search hypers:"
            .format(grid.best_params_, grid.best_score_), self.mean_confidence_interval(grid, x_test, y_test))

        print(grid.cv_results_)

        curve_name = short_name + " " + dataset
        plot_roc_curve(grid.best_estimator_, x_test, y_test, name=curve_name)
        plt.savefig('../plots/' + curve_name)
        # plt.show()

    def perform_train_full(self):
        print('\n############################# Full datasets #############################')
        for i in self.dataset_full:
            print('----------------Training {}----------------'.format(i))
            dataset = [self.dict_dataframes['df_full_' + i]]
            metadata = [self.dict_dataframes['pbr_res_' + i]]
            name = i

            self.perform_feature_selection(dataset, metadata)

            self.train_SGDClassifier(dataset, metadata, name)
            self.train_logistic_regression(dataset, metadata, name)
            self.train_SVC(dataset, metadata, name)
            # self.train_random_forest(dataset, metadata, name)
            # self.train_GBTC(dataset, metadata, name)
        return self

    def perform_train_gwas(self):
        print('\n############################# GWAS datasets #############################')
        for i in self.datasets_gwas:
            print('----------------Training {}----------------'.format(i))
            dataset = [self.dict_dataframes['df_full_' + i],
                       self.dict_dataframes['df_gwas_' + i]]
            metadata = [self.dict_dataframes['pbr_res_' + i]]
            name = i

            self.perform_feature_selection(dataset, metadata)

            self.train_SGDClassifier(dataset, metadata, name)
            self.train_logistic_regression(dataset, metadata, name)
            self.train_SVC(dataset, metadata, name)
            # self.train_random_forest(dataset, metadata, name)
            # self.train_GBTC(dataset, metadata, name)
        return self

    def perform_feature_selection(self, dataset, metadata):
        encoder = LabelEncoder()
        isolate_column = dataset[0].columns[0]
        dataset[0][isolate_column] = encoder.fit_transform(dataset[0][isolate_column])

        estimator = RandomForestClassifier(random_state=self.SEED)
        # estimator = SVR(kernel="linear", random_state=self.SEED)
        selector = RFECV(estimator, step=1, cv=StratifiedKFold(10), scoring='roc_auc')
        selector = selector.fit(dataset[0], metadata[0].values.ravel())

        selected_features = dataset[0].columns[selector.support_]
        dataset[0] = dataset[0].filter(selected_features)
        return self

    def train_SGDClassifier(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = self.perform_train_test_split(dataset, metadata)

        parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10], 'loss': ['hinge', 'log'],
                      'penalty': ['l1', 'l2']}
        model = SGDClassifier(random_state=self.SEED, max_iter=10000)
        grid = GridSearchCV(model, parameters, cv=10, scoring='accuracy', return_train_score=True)
        grid.fit(x_train, y_train)

        model_info = ["Stochastic Gradient Descent Classifier", x_test, y_test, "SGDC"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2], name, model_info[3])

        return self

    def train_logistic_regression(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = self.perform_train_test_split(dataset, metadata)

        parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l2']}
        model = LogisticRegression(random_state=self.SEED, max_iter=10000)
        grid = GridSearchCV(model, parameters, cv=10, scoring='accuracy', return_train_score=True)
        grid.fit(x_train, y_train)

        model_info = ["Logistic Regression", x_test, y_test, "LR"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2], name, model_info[3])

        return self

    def train_SVC(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = self.perform_train_test_split(dataset, metadata)

        parameters = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10],
                      'kernel': ['rbf', 'poly', 'sigmoid']}
        model = SVC(random_state=self.SEED, max_iter=10000)
        grid = GridSearchCV(model, parameters, cv=10, scoring='accuracy', return_train_score=True)
        grid.fit(x_train, y_train)

        model_info = ["Support Vectors Classifier", x_test, y_test, "SVC"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2], name, model_info[3])

        return self

    def train_GBTC(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = self.perform_train_test_split(dataset, metadata)

        parameters = {'learning_rate': [0.001, 0.01, 0.1], 'n_estimators': [50, 100, 200, 300, 400, 500],
                      'max_depth': [3, 5, 10, 12, 15]}
        model = GradientBoostingClassifier(random_state=self.SEED)
        grid = GridSearchCV(model, parameters, cv=10, scoring='accuracy', return_train_score=True)
        grid.fit(x_train, y_train)

        model_info = ["Gradient Boosting Classifier", x_test, y_test, "GBTC"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2], name, model_info[3])
        return self

    def train_random_forest(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = self.perform_train_test_split(dataset, metadata)

        parameters = {'n_estimators': [50, 100, 200, 400], 'max_features': ['sqrt', 'log2'],
                      'max_depth': [10, 12, 15, 20], 'criterion': ['gini', 'entropy']}
        model = RandomForestClassifier(random_state=self.SEED)
        grid = GridSearchCV(model, parameters, cv=10, scoring='accuracy', return_train_score=True)
        grid.fit(x_train, y_train)

        model_info = ["Random Forest", x_test, y_test, "RF"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2], name, model_info[3])

        return self

    def perform_XGB(self, dataset, metadata, name):
        # x_train, x_test, y_train, y_test = self.perform_train_test_split(dataset, metadata)
        #
        # train_matrix = xgb.DMatrix(x_train, y_trai n)
        # test_matrix = xgb.DMatrix(x_test, y_test)
        #
        # model = xgb.train()

        return self

    def main(self):
        self.read_datasets()

        if self.dataset_full:
            self.perform_train_full()
        if self.datasets_gwas:
            self.perform_train_gwas()


if __name__ == "__main__":
    # Uncomment to genetare log.txt
    # stdoutOrigin = sys.stdout
    # sys.stdout = open("log_fs_SVN_50estimators.txt", "w")

    df01 = ['dataset_1_', 'dataset_2_', 'dataset_3_']
    df02 = ['dataset_19_', 'dataset_20_', 'dataset_21_']

    PR = PredictPRinKP(datasets_full=df01, datasets_gwas=df02)
    PR.main()

    # sys.stdout.close()
    # sys.stdout = stdoutOrigin

    print("finished model training")
